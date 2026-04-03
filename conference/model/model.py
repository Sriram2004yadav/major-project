import torch
import torch.nn as nn
from torch import Tensor
import einops

from typing import Tuple, Sequence
from pathlib import Path

from utils.experiment_manager import CfgNode
from model import unet, modules


class ContUrbanCDModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(ContUrbanCDModel, self).__init__()

        # Attributes — identical to original
        self.cfg      = cfg
        self.c        = cfg.MODEL.IN_CHANNELS
        self.d_out    = cfg.MODEL.OUT_CHANNELS
        self.h = self.w = cfg.AUGMENTATION.CROP_SIZE
        self.t        = cfg.DATALOADER.TIMESERIES_LENGTH
        self.topology = cfg.MODEL.TOPOLOGY

        # ConvNet layers — identical to original
        self.inc         = unet.InConv(self.c, self.topology[0], unet.DoubleConv)
        self.encoder     = unet.Encoder(cfg)
        self.decoder_seg = unet.Decoder(cfg)
        self.outc_seg    = unet.OutConv(self.topology[0], self.d_out)
        self.decoder_ch  = unet.Decoder(cfg)
        self.outc_ch     = unet.OutConv(self.topology[0], self.d_out)

        # TFR modules — identical to original
        tfr_modules      = []
        transformer_dims = [self.topology[-1]] + list(self.topology[::-1])
        for i, d_model in enumerate(transformer_dims):
            tfr_module = modules.TFRModule(
                t          = self.t,
                d_model    = d_model,
                n_heads    = cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS,
                d_hid      = self.topology[0] * 4,
                activation = cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION,
                n_layers   = cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS,
            )
            tfr_modules.append(tfr_module)
        self.tfr_modules = nn.ModuleList(tfr_modules)

        # CF module — identical to original
        self.cf_module = modules.CFModule()

        # MTI module — identical to original
        self.mti_module = modules.MTIModule()

        # NEW: Bidirectional calibrator
        # Calibrates decoder outputs before MTI receives them.
        # Trained end-to-end via loss_cal in train.py.
        self.calibrator = modules.BiConvGRUCalibrator(
            in_ch     = 2,
            hidden_ch = cfg.MODEL.REFINER.HIDDEN_CH,
        )

    def forward(
        self,
        x:     Tensor,
        edges: Sequence[Tuple[int, int]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            logits_ch         : (B, N, 1, H, W)
            logits_seg        : (B, T, 1, H, W)
            calibrated_logits : (B, T, 1, H, W)  — NEW third output
        """
        B, T, _, H, W = x.size()

        # Feature extraction — identical to original
        x        = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        features = self.encoder(self.inc(x))
        features = [einops.rearrange(f_s, '(b t) f h w -> b t f h w', b=B) for f_s in features]

        # TFR modules — identical to original
        for i, tfr_module in enumerate(self.tfr_modules):
            f          = features[i]
            f_refined  = tfr_module(f)
            features[i] = f_refined

        # CF module — identical to original
        features_ch = self.cf_module(features, edges)
        features_ch = [einops.rearrange(f, 'n b c h w -> (b n) c h w') for f in features_ch]

        # Segmentation decoder — identical to original
        features_seg = [einops.rearrange(f, 'b t c h w -> (b t) c h w') for f in features]
        logits_seg   = self.outc_seg(self.decoder_seg(features_seg))
        logits_seg   = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=B)

        # Change decoder — identical to original
        logits_ch = self.outc_ch(self.decoder_ch(features_ch))
        logits_ch = einops.rearrange(logits_ch, '(b n) c h w -> b n c h w', n=len(edges))

        # NEW: calibrate outputs using bidirectional temporal context
        calibrated_logits = self.calibrator(logits_seg, logits_ch, edges)

        return logits_ch, logits_seg, calibrated_logits

    def inference(
        self,
        x:     Tensor,
        edges: Sequence[Tuple[int, int]]
    ) -> Tensor:
        """
        Inference path — used by eval.py and inference.py (unchanged).

        KEY CHANGE vs original:
          Original: sigmoid(logits_seg)    → MTI
          Ours:     sigmoid(calibrated_logits) → MTI

          MTI now receives calibrated, temporally coherent probabilities
          instead of raw decoder outputs → belief propagation is meaningful.
        """
        logits_ch, logits_seg, calibrated_logits = self.forward(x, edges)
        o_ch  = torch.sigmoid(logits_ch).detach()
        o_seg = torch.sigmoid(calibrated_logits).detach()  # calibrated
        o_seg = self.mti_module(o_ch, o_seg, edges)
        return o_seg


# Helpers — identical to original
def init_model(cfg: CfgNode) -> nn.Module:
    net = ContUrbanCDModel(cfg)
    return torch.nn.DataParallel(net)


def save_model(network: nn.Module, epoch: float, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
    save_file.parent.mkdir(exist_ok=True)
    checkpoint = {'epoch': epoch, 'weights': network.state_dict()}
    torch.save(checkpoint, save_file)


def load_model(cfg: CfgNode, device: torch.device) -> nn.Module:
    net      = init_model(cfg)
    net.to(device)
    net_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
    checkpoint = torch.load(net_file, map_location=device)
    net.load_state_dict(checkpoint['weights'])
    return net


def power_jaccard_loss(
    input: Tensor,
    target: Tensor,
    disable_sigmoid: bool = False
) -> Tensor:
    input_sigmoid = torch.sigmoid(input) if not disable_sigmoid else input
    eps           = 1e-6
    iflat         = input_sigmoid.flatten()
    tflat         = target.flatten()
    intersection  = (iflat * tflat).sum()
    denom         = (iflat ** 2 + tflat ** 2).sum() - (iflat * tflat).sum() + eps
    return 1 - (intersection / denom)