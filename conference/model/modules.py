import torch
import torch.nn as nn
from torch import Tensor
import einops
import numpy as np

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

from typing import Tuple, Sequence, Callable
from joblib import Parallel, delayed

import os
os.environ["PYTHONWARNINGS"] = "ignore"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# ORIGINAL AUTHORS CODE — NOT MODIFIED
# ==============================================================================

class TFRModule(nn.Module):
    def __init__(self, t: int, d_model: int, n_heads: int, d_hid: int, activation: str, n_layers: int):
        super().__init__()
        self.register_buffer('temporal_encodings', self.get_relative_encodings(t, d_model), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_hid, batch_first=True,
            activation=activation
        )
        self.temporal_feature_refinement = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, features: Tensor) -> Tensor:
        B, T, D, H, W = features.size()
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)
        features_hat = self.temporal_feature_refinement(tokens)
        features_hat = einops.rearrange(features_hat, '(B H W) T D -> B T D H W', B=B, H=H)
        return features_hat

    @staticmethod
    def get_relative_encodings(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


class CFModule(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(features: Sequence[Tensor], edges: Sequence[Tuple[int, int]]) -> Sequence[Tensor]:
        features_ch = []
        for feature in features:
            B, T, _, H, W = feature.size()
            feature_ch = []
            for t1, t2 in edges:
                feature_ch.append(feature[:, t2] - feature[:, t1])
            feature_ch = torch.stack(feature_ch)
            features_ch.append(feature_ch)
        return features_ch


class MTIModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, o_ch: Tensor, o_seg: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:
        B, T, _, H, W = o_seg.size()
        process_pixel = self.markov_network(T, edges)
        o_ch  = einops.rearrange(o_ch,  'B N C H W -> (B H W) N C').cpu().numpy()
        o_seg = einops.rearrange(o_seg, 'B T C H W -> (B H W) T C').cpu().numpy()
        o_seg_mrf = Parallel(n_jobs=-1)(
            delayed(process_pixel)(p_seg, p_ch) for p_seg, p_ch in zip(o_seg, o_ch)
        )
        o_seg_mrf = torch.Tensor(o_seg_mrf)
        o_seg_mrf = einops.rearrange(o_seg_mrf, '(B H W) (T C) -> B T C H W', H=H, W=W, T=T)
        return o_seg_mrf

    @staticmethod
    def markov_network(n: int, edges: Sequence[Tuple[int, int]]) -> Callable:
        def process_pixel(y_hat_seg: Sequence[float], y_hat_ch: Sequence[float]):
            model = MarkovNetwork()
            for t in range(len(y_hat_seg)):
                model.add_node(f'N{t}')
                urban_value = float(y_hat_seg[t].item() if hasattr(y_hat_seg[t], 'item') else y_hat_seg[t].flat[0])
                factor = DiscreteFactor([f'N{t}'], cardinality=[2], values=[1 - urban_value, urban_value])
                model.add_factors(factor)
            for t in range(n - 1):
                model.add_edge(f'N{t}', f'N{t + 1}')
            for i, (t1, t2) in enumerate(edges):
                model.add_edge(f'N{t1}', f'N{t2}')
                change_value = float(y_hat_ch[i].item() if hasattr(y_hat_ch[i], 'item') else y_hat_ch[i].flat[0])
                edge_values = [1 - change_value, change_value, change_value, 1 - change_value]
                factor = DiscreteFactor([f'N{t1}', f'N{t2}'], cardinality=[2, 2], values=edge_values)
                model.add_factors(factor)
            bp = BeliefPropagation(model)
            state = bp.map_query()
            states_list = [state[f'N{t}'] for t in range(n)]
            return states_list
        return process_pixel


# ==============================================================================
# NEW CONTRIBUTION 1 — BiConvGRUCalibrator
#
# Addresses paper's stated limitation (Section VI Discussion):
#   "outputs of deep networks may not be well-calibrated"
#
# Position: sits BEFORE MTI, does NOT replace it.
#   decoder outputs → BiConvGRUCalibrator → MTI (unchanged)
#
# Why bidirectional:
#   Forward-only GRU cannot use t+1 to correct t.
#   All timestamps are available at once (offline satellite setting),
#   so bidirectional is algorithmically justified.
#   A building visible at t=4 should raise calibration confidence at t=3.
#
# Training:
#   Supervised with y_seg via loss_cal in train.py.
#   Gradients flow end-to-end through both directions.
#   Network is incentivised to produce well-calibrated outputs for MTI.
# ==============================================================================

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 2 * hidden_dim,
            kernel_size, padding=pad, bias=True
        )
        self.conv_can = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size, padding=pad, bias=True
        )

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        combined = torch.cat([x, h], dim=1)
        gates    = self.conv_gates(combined)
        r        = torch.sigmoid(gates[:, :self.hidden_dim])
        z        = torch.sigmoid(gates[:, self.hidden_dim:])
        n        = torch.tanh(self.conv_can(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * n

    def init_hidden(self, B: int, H: int, W: int, device) -> Tensor:
        return torch.zeros(B, self.hidden_dim, H, W, device=device)


class BiConvGRUCalibrator(nn.Module):
    def __init__(self, in_ch: int = 2, hidden_ch: int = 32):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.fwd_cell  = ConvGRUCell(in_ch, hidden_ch)
        self.bwd_cell  = ConvGRUCell(in_ch, hidden_ch)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.out_conv  = nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True)

    def _align_change(
        self,
        logits_seg: Tensor,
        logits_ch:  Tensor,
        edges:      Sequence[Tuple[int, int]],
        T:          int
    ) -> Tensor:
        """
        Extracts adjacent-pair change probs from the full edge list.
        Works correctly for ALL edge types (adjacent / cyclic / dense).

        Example with dense edges T=5:
          edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
          We extract only: (0,1)→idx0, (1,2)→idx4, (2,3)→idx7, (3,4)→idx9
          This fixes the critical bug where dense edges always gave zeros.
        """
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        ch_aligned  = torch.zeros_like(logits_seg)  # (B, T, 1, H, W)

        for t in range(1, T):
            if (t - 1, t) in edge_to_idx:
                idx              = edge_to_idx[(t - 1, t)]
                ch_aligned[:, t] = torch.sigmoid(logits_ch[:, idx])

        return ch_aligned

    def forward(
        self,
        logits_seg: Tensor,
        logits_ch:  Tensor,
        edges:      Sequence[Tuple[int, int]]
    ) -> Tensor:
        """
        Args:
            logits_seg : (B, T, 1, H, W)  raw segmentation logits from decoder
            logits_ch  : (B, N, 1, H, W)  raw change logits from decoder
            edges      : list of (t1, t2) tuples
        Returns:
            calibrated_logits : (B, T, 1, H, W)
        """
        B, T, _, H, W = logits_seg.shape

        seg_probs  = torch.sigmoid(logits_seg)
        ch_aligned = self._align_change(logits_seg, logits_ch, edges, T)
        inputs     = torch.cat([seg_probs, ch_aligned], dim=2)  # (B, T, 2, H, W)

        # Forward pass
        h_fwd    = self.fwd_cell.init_hidden(B, H, W, logits_seg.device)
        fwd_list = []
        for t in range(T):
            h_fwd = self.fwd_cell(inputs[:, t], h_fwd)
            fwd_list.append(h_fwd)

        # Backward pass
        h_bwd    = self.bwd_cell.init_hidden(B, H, W, logits_seg.device)
        bwd_list = [None] * T
        for t in reversed(range(T)):
            h_bwd       = self.bwd_cell(inputs[:, t], h_bwd)
            bwd_list[t] = h_bwd

        # Fuse and project
        calibrated = []
        for t in range(T):
            fused = torch.cat([fwd_list[t], bwd_list[t]], dim=1)
            fused = self.fuse_conv(fused)
            calibrated.append(self.out_conv(fused))

        return torch.stack(calibrated, dim=1)  # (B, T, 1, H, W)


# ==============================================================================
# NEW CONTRIBUTION 2 — TemporalConsistencyLoss
#
# Addresses the training gap the paper leaves open.
# The paper enforces temporal consistency ONLY at inference via MTI.
# The network is never trained with any constraint that makes change
# predictions coherent across the full time series.
#
# For every valid triplet (a, b, c) where a < b < c:
#   p(a→c)  should equal  softXOR( p(a→b), p(b→c) )
#          = p(a→b) + p(b→c) − 2·p(a→b)·p(b→c)
#
# This is because change(a,c) = change(a,b) XOR change(b,c) must hold
# for any binary state (building present/absent).
#
# With T=5 dense edges: C(5,3) = 10 valid triplets per training sample.
# ==============================================================================

import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, lambda_tcl: float = 0.1):
        super().__init__()
        self.lambda_tcl = lambda_tcl

    def forward(
        self,
        logits_ch: Tensor,
        edges:     Sequence[Tuple[int, int]],
        T:         int
    ) -> Tensor:
        probs       = torch.sigmoid(logits_ch)
        edge_to_idx = {e: i for i, e in enumerate(edges)}

        total = torch.tensor(0.0, device=logits_ch.device)
        count = 0

        for a in range(T):
            for b in range(a + 1, T):
                for c in range(b + 1, T):
                    if (a, b) not in edge_to_idx: continue
                    if (b, c) not in edge_to_idx: continue
                    if (a, c) not in edge_to_idx: continue

                    p_ab     = probs[:, edge_to_idx[(a, b)]]
                    p_bc     = probs[:, edge_to_idx[(b, c)]]
                    p_ac     = probs[:, edge_to_idx[(a, c)]]
                    expected = p_ab + p_bc - 2.0 * p_ab * p_bc

                    total = total + F.mse_loss(p_ac, expected.detach())
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits_ch.device)

        return self.lambda_tcl * (total / count)