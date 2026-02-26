import sys
import os
import torch
from collections import OrderedDict
from utils import evaluation, experiment_manager, parsers
from model.model import ContUrbanCDModel

def evaluate_checkpoint():
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 🔍 Starting Evaluation on {device} ===")

    # Initialize Clean Model
    net = ContUrbanCDModel(cfg)
    net = torch.nn.DataParallel(net)
    net.to(device)

    checkpoint_path = args.resume 
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("❌ Error: Valid .pt file required.")
        sys.exit(1)

    print(f"📥 Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device).get('weights', torch.load(checkpoint_path))
    
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    curr_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    
    # Universal Loader Match
    matched_dict = {k: v for k, v in clean_dict.items() if k in curr_dict and v.shape == curr_dict[k].shape}
    
    target_model = net.module if hasattr(net, 'module') else net
    target_model.load_state_dict(matched_dict, strict=False)
    print(f"✅ Loaded {len(matched_dict)} layers.")

    net.eval()
    with torch.no_grad():
        print("\n🚀 Running Evaluation...")
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', 0, 0)
    print(f"\n🏆 Final Total F1: {f1_val:.4f}")

if __name__ == '__main__':
    evaluate_checkpoint()