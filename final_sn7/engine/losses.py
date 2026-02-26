import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSN7Loss(nn.Module):
    """
    Combines BCE (for sharp edges) and Power Jaccard (for area overlap).
    """
    def __init__(self, pos_weight=2.0, p=1.5, bce_weight=0.5, jaccard_weight=0.5):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.p = p
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight

    def power_jaccard(self, logits, target):
        probs = torch.sigmoid(logits)
        iflat = probs.flatten()
        tflat = target.flatten().float()
        
        pos_w = torch.tensor([self.pos_weight_val], device=logits.device)
        
        tp = (iflat * tflat).sum()
        fp = (iflat * (1 - tflat)).sum()
        fn = ((1 - iflat) * tflat).sum()
        
        num = tp * pos_w
        den = num + fp + (fn * pos_w) + 1e-6
        return 1 - (num / den)**self.p

    def forward(self, logits, target):
        pos_w = torch.tensor([self.pos_weight_val], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_w)
        jac_loss = self.power_jaccard(logits, target)
        
        return (self.bce_weight * bce_loss) + (self.jaccard_weight * jac_loss)