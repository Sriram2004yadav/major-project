import abc

import torch
from torch.utils import data as torch_data
from torch import Tensor

from utils import datasets, helpers
import wandb

EPS = 10e-05


class AbstractMeasurer(abc.ABC):
    def __init__(self, threshold: float = 0.5, name: str = None):

        self.threshold = threshold
        self.name = name

        # urban mapping
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError("add_sample method must be implemented in the subclass.")

    def _update_metrics(self, y: Tensor, y_hat: Tensor, attr_name: str, mask: Tensor = None):
        y = y.bool()
        y_hat = y_hat > self.threshold

        tp_attr = f'TP_{attr_name}'
        tn_attr = f'TN_{attr_name}'
        fp_attr = f'FP_{attr_name}'
        fn_attr = f'FN_{attr_name}'

        tp = (y & y_hat).float()
        tn = (~y & ~y_hat).float()
        fp = (y_hat & ~y).float()
        fn = (~y_hat & y).float()

        if mask is not None:
            tp[mask] = float('nan')
            tn[mask] = float('nan')
            fp[mask] = float('nan')
            fn[mask] = float('nan')

        setattr(self, tp_attr, getattr(self, tp_attr) + torch.nansum(tp).float().item())
        setattr(self, tn_attr, getattr(self, tn_attr) + torch.nansum(tn).float().item())
        setattr(self, fp_attr, getattr(self, fp_attr) + torch.nansum(fp).float().item())
        setattr(self, fn_attr, getattr(self, fn_attr) + torch.nansum(fn).float().item())


class MultiTaskMeasurer(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_seg: Tensor, y_hat_seg: Tensor, y_ch: Tensor, y_hat_ch: Tensor, mask: Tensor = None):

        # urban mapping
        if y_seg is not None:
            self._update_metrics(y_seg, y_hat_seg, 'seg_cont', mask)
            self._update_metrics(y_seg[:, [0, -1]], y_hat_seg[:, [0, -1]], 'seg_fl', mask)

        # urban change
        if y_hat_ch.size(1) > 1:
            self._update_metrics(y_ch[:, :-1], y_hat_ch[:, :-1], 'ch_cont', mask)
        self._update_metrics(y_ch[:, -1], y_hat_ch[:, -1], 'ch_fl', mask)


def run_quantitative_evaluation(net, cfg, device, run_type: str, enable_mti: bool = False,
                                mti_edge_setting: str = 'dense') -> MultiTaskMeasurer:
    tile_size = cfg.AUGMENTATION.CROP_SIZE
    ds = datasets.create_eval_dataset(cfg, run_type, tiling=tile_size)

    net.to(device)
    net.eval()

    m = MultiTaskMeasurer()
    # --- START MODIFICATION ---
    # Use the edge type from the config, not hardcoded 'cyclic'
    edges_for_eval = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.MODEL.EDGE_TYPE)
    # --- END MODIFICATION ---
    edges_mti = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, mti_edge_setting)

    batch_size = 1 if enable_mti else cfg.TRAINER.BATCH_SIZE
    dataloader = torch_data.DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

    for step, item in enumerate(dataloader):
        # --- START MODIFICATION ---
        x = item['x'].to(device)

        if cfg.DATASET.NAME == 'tscd':
            y_seg = None # No segmentation labels for TSCD
            y_ch = item['y_ch'].to(device)
        else:
            y_seg = item['y']
            # --- MODIFICATION ---
            y_ch = helpers.get_ch(y_seg, edges_for_eval)
        
        with torch.no_grad():
            if enable_mti:
                o_seg = net.module.inference(x, edges_mti)
                # --- MODIFICATION ---
                o_ch = helpers.get_ch(o_seg, edges_for_eval)
            else:
                # --- MODIFICATION ---
                logits_ch, logits_seg = net(x, edges_for_eval)
                o_ch, o_seg = torch.sigmoid(logits_ch).detach(), torch.sigmoid(logits_seg).detach()
        
        # For TSCD, y_seg will be None, and m.add_sample will safely skip it
        m.add_sample(y_seg.cpu() if y_seg is not None else None, o_seg.cpu(), y_ch.cpu(), o_ch.cpu())
        # --- END MODIFICATION ---

    return m


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    m = run_quantitative_evaluation(net, cfg, device, run_type)

    # --- START MODIFICATION ---
    if cfg.DATASET.NAME == 'tscd':
        # For TSCD, we only care about the change F1 score
        f1_ch_cont = f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
        f1_ch_fl = f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
        f1 = (f1_ch_cont + f1_ch_fl) / 2 # The validation F1 is the average change F1
        
        wandb.log({
            f'{run_type} f1': f1,
            f'{run_type} f1 ch cont': f1_ch_cont,
            f'{run_type} f1 ch fl': f1_ch_fl,
            'step': step, 'epoch': epoch,
        })
    
    else:
        # Original logic for SN7 and WUSU
        f1_seg_cont = f1_score(m.TP_seg_cont, m.FP_seg_cont, m.FN_seg_cont)
        f1_seg_fl = f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl)
        f1_ch_cont = f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
        f1_ch_fl = f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
        f1 = (f1_seg_cont + f1_seg_fl + f1_ch_cont + f1_ch_fl) / 4
    
        wandb.log({
            f'{run_type} f1': f1,
            f'{run_type} f1 seg cont': f1_seg_cont,
            f'{run_type} f1 seg fl': f1_seg_fl,
            f'{run_type} f1 ch cont': f1_ch_cont,
            f'{run_type} f1 ch fl': f1_ch_fl,
            'step': step, 'epoch': epoch,
        })
    # --- END MODIFICATION ---

    return f1


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp + EPS)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn + EPS)


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * p * r) / (p + r + EPS)


def iou(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp + fn + EPS)