import torch
import torch.nn as nn
from torch import Tensor
import einops
from typing import Tuple, Sequence
from pathlib import Path

from model import unet, modules

class ContUrbanCDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        topo = cfg.MODEL.TOPOLOGY
        t_len = cfg.DATALOADER.TIMESERIES_LENGTH
        
        # ConvNet layers
        self.inc = unet.InConv(cfg.MODEL.IN_CHANNELS, topo[0], unet.DoubleConv)
        self.encoder = unet.Encoder(cfg)
        self.decoder_seg = unet.Decoder(cfg)
        self.outc_seg = unet.OutConv(topo[0], cfg.MODEL.OUT_CHANNELS)
        self.decoder_ch = unet.Decoder(cfg)
        self.outc_ch = unet.OutConv(topo[0], cfg.MODEL.OUT_CHANNELS)

        # Temporal feature refinement (TFR) modules
        tfr_dims = [topo[-1]] + list(topo[::-1])
        self.tfr_modules = nn.ModuleList([
            modules.TFRModule(
                t=t_len, d_model=d,
                n_heads=cfg.MODEL.TRANSFORMER_PARAMS.N_HEADS,
                d_hid=topo[0] * 4,
                activation=cfg.MODEL.TRANSFORMER_PARAMS.ACTIVATION,
                n_layers=cfg.MODEL.TRANSFORMER_PARAMS.N_LAYERS
            ) for d in tfr_dims
        ])

        # Change & Refinement
        self.cf_module = modules.CFModule()
        self.neural_refiner = modules.NeuralRefiner(
            in_ch=2, 
            hidden_ch=cfg.MODEL.REFINER.HIDDEN_CH
        ) 

    def forward(self, x: Tensor, edges: Sequence[Tuple[int, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, _, H, W = x.size()

        # Siamese ConvNet encoder
        x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        features = self.encoder(self.inc(x_flat))
        features = [einops.rearrange(f_s, '(b t) f h w -> b t f h w', b=B) for f_s in features]

        # TFR modules
        for i, tfr in enumerate(self.tfr_modules):
            features[i] = tfr(features[i])

        # Change feature maps
        features_ch = self.cf_module(features, edges)
        features_ch = [einops.rearrange(f, 'n b c h w -> (b n) c h w') for f in features_ch]
        logits_ch = self.outc_ch(self.decoder_ch(features_ch))
        logits_ch = einops.rearrange(logits_ch, '(b n) c h w -> b n c h w', n=len(edges))

        # Raw Segmentation
        features_seg = [einops.rearrange(f, 'b t c h w -> (b t) c h w') for f in features]
        logits_seg = self.outc_seg(self.decoder_seg(features_seg))
        logits_seg = einops.rearrange(logits_seg, '(b t) c h w -> b t c h w', b=B)

        # Refined Seg
        refined_seg = self.neural_refiner(logits_seg, logits_ch)

        return logits_ch, logits_seg, refined_seg

def save_model(network: nn.Module, epoch: float, cfg):
    save_file = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'
    save_file.parent.mkdir(exist_ok=True, parents=True)
    checkpoint = {'epoch': epoch, 'weights': network.state_dict()}
    torch.save(checkpoint, save_file)