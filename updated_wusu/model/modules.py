import torch
import torch.nn as nn
from torch import Tensor
import einops
import numpy as np
from typing import Tuple, Sequence

# --- NOTE: We removed pgmpy imports because we replaced MTI with NeuralRefiner ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TFRModule(nn.Module):
    def __init__(self, t: int, d_model: int, n_heads: int, d_hid: int, activation: str, n_layers: int):
        super().__init__()
        # Generate relative temporal encodings
        self.register_buffer('temporal_encodings', self.get_relative_encodings(t, d_model), persistent=False)

        # Define a transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_hid, batch_first=True,
            activation=activation
        )
        # Create module
        self.temporal_feature_refinement = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, features: Tensor) -> Tensor:
        B, T, D, H, W = features.size()

        # Reshape to tokens
        tokens = einops.rearrange(features, 'B T D H W -> (B H W) T D')

        # Adding relative temporal encodings
        tokens = tokens + self.temporal_encodings.repeat(B * H * W, 1, 1)

        # Feature refinement with self-attention
        features_hat = self.temporal_feature_refinement(tokens)

        # Reshape to original shape
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
        # compute urban change detection features
        features_ch = []
        for feature in features:
            B, T, _, H, W = feature.size()
            feature_ch = []
            for t1, t2 in edges:
                feature_ch.append(feature[:, t2] - feature[:, t1])
            # n: number of combinations
            feature_ch = torch.stack(feature_ch)
            features_ch.append(feature_ch)
        return features_ch


# --- SOTA NEURAL REFINER (Replaces MTIModule) ---

class ConvGRUCell(nn.Module):
    """
    A learnable Spatio-Temporal consistency unit.
    It replaces the static Markov transition rules with learnable gates.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # Gates: Reset (r) and Update (z)
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, 
                                    kernel_size, padding=padding, bias=bias)
        # Candidate memory (h_tilde)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 
                                  kernel_size, padding=padding, bias=bias)

    def forward(self, input_tensor, h_cur):
        # input_tensor: (B, C, H, W)
        # h_cur: (B, Hidden, H, W)
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        
        combined_new = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined_new)
        cnm = torch.tanh(cc_cnm)
        
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class NeuralRefiner(nn.Module):
    """
    SOTA REPLACEMENT for MTIModule.
    Takes raw Segmentation and Change probabilities and refines them
    using temporal memory.
    """
    def __init__(self, in_ch=2, hidden_ch=16):
        super().__init__()
        # Input channels = 1 (Seg) + 1 (Change) = 2
        self.rnn = ConvGRUCell(input_dim=in_ch, hidden_dim=hidden_ch, kernel_size=3, bias=True)
        self.out_conv = nn.Conv2d(hidden_ch, 1, kernel_size=1) # Final binary prediction

    def forward(self, logits_seg, logits_ch):
        # logits_seg: (B, T, 1, H, W)
        # logits_ch: (B, N_edges, 1, H, W)
        
        B, T, C, H, W = logits_seg.size()
        
        # 1. Align Change maps to Time. 
        zero_pad = torch.zeros_like(logits_seg[:, 0:1])
        
        if logits_ch.size(1) == T - 1:
             ch_aligned = torch.cat([zero_pad, logits_ch], dim=1)
        else:
            ch_aligned = torch.zeros_like(logits_seg)

        # 2. Initialize Memory (Hidden State)
        h = torch.zeros(B, self.rnn.hidden_dim, H, W).to(logits_seg.device)
        
        refined_outputs = []
        
        # 3. Process the Timeline
        for t in range(T):
            input_t = torch.cat([logits_seg[:, t], ch_aligned[:, t]], dim=1)
            h = self.rnn(input_t, h)
            out_t = self.out_conv(h)
            refined_outputs.append(out_t)
            
        # Stack back to (B, T, 1, H, W)
        return torch.stack(refined_outputs, dim=1)