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
        # Metrics storage
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

    def _update_metrics(self, y: Tensor, y_hat: Tensor, attr_name: str, mask: Tensor = None):
        y = y.bool()
        y_hat = y_hat > self.threshold
        tp_attr, tn_attr = f'TP_{attr_name}', f'TN_{attr_name}'
        fp_attr, fn_attr = f'FP_{attr_name}', f'FN_{attr_name}'

        tp = (y & y_hat).float()
        tn = (~y & ~y_hat).float()
        fp = (y_hat & ~y).float()
        fn = (~y_hat & y).float()

        if mask is not None:
            tp[mask] = tn[mask] = fp[mask] = fn[mask] = float('nan')

        setattr(self, tp_attr, getattr(self, tp_attr) + torch.nansum(tp).float().item())
        setattr(self, tn_attr, getattr(self, tn_attr) + torch.nansum(tn).float().item())
        setattr(self, fp_attr, getattr(self, fp_attr) + torch.nansum(fp).float().item())
        setattr(self, fn_attr, getattr(self, fn_attr) + torch.nansum(fn).float().item())

class MultiTaskMeasurer(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_seg: Tensor, y_hat_seg: Tensor, y_ch: Tensor, y_hat_ch: Tensor, mask: Tensor = None):
        if y_seg is not None:
            self._update_metrics(y_seg, y_hat_seg, 'seg_cont', mask)
            self._update_metrics(y_seg[:, [0, -1]], y_hat_seg[:, [0, -1]], 'seg_fl', mask)
        if y_hat_ch.size(1) > 1:
            self._update_metrics(y_ch[:, :-1], y_hat_ch[:, :-1], 'ch_cont', mask)
        self._update_metrics(y_ch[:, -1], y_hat_ch[:, -1], 'ch_fl', mask)

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

def run_quantitative_evaluation(net, cfg, device, run_type: str, enable_mti: bool = False, mti_edge_setting: str = 'dense') -> MultiTaskMeasurer:
    tile_size = cfg.AUGMENTATION.CROP_SIZE
    ds = datasets.create_eval_dataset(cfg, run_type, tiling=tile_size)
    net.to(device)
    net.eval()
    m = MultiTaskMeasurer()
    # Updated to pull from DATALOADER instead of MODEL
    edges_for_eval = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.DATALOADER.EDGE_TYPE)
    
    # Use num_workers=0 for stability during validation
    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)

    for step, item in enumerate(dataloader):
        x = item['x'].to(device)
        y_seg = item['y'].to(device)
        y_ch = helpers.get_ch(y_seg, edges_for_eval)
        
        with torch.no_grad():
            # In your new_codebase, the model returns (logits_ch, logits_seg, refined_seg)
            logits_ch, logits_seg, refined_seg = net(x, edges_for_eval)
            o_ch = torch.sigmoid(logits_ch).detach()
            o_seg = torch.sigmoid(refined_seg).detach()
        
        m.add_sample(y_seg.cpu(), o_seg.cpu(), y_ch.cpu(), o_ch.cpu())
    return m

def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    # Now correctly defined above
    m = run_quantitative_evaluation(net, cfg, device, run_type)
    
    f1_seg_cont = f1_score(m.TP_seg_cont, m.FP_seg_cont, m.FN_seg_cont)
    f1_seg_fl = f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl)
    f1_ch_cont = f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
    f1_ch_fl = f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
    
    f1 = (f1_seg_cont + f1_seg_fl + f1_ch_cont + f1_ch_fl) / 4
    
    print(f"--- VALIDATION RESULTS (Epoch {epoch}) ---")
    print(f"F1 Total: {f1:.4f} | Seg: {f1_seg_cont:.4f} | Change: {f1_ch_cont:.4f}")

    # Safely log to wandb only if initialized
    if wandb.run is not None:
        wandb.log({
            f'{run_type} f1': f1,
            f'{run_type} f1 seg cont': f1_seg_cont,
            f'{run_type} f1 seg fl': f1_seg_fl,
            f'{run_type} f1 ch cont': f1_ch_cont,
            f'{run_type} f1 ch fl': f1_ch_fl,
            'step': step, 'epoch': epoch,
        })

    return f1