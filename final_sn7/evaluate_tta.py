import sys
import os
import torch
from torch.utils import data as torch_data

from utils import evaluation, experiment_manager, parsers, datasets, helpers
from model.model import ContUrbanCDModel

def evaluate_with_tta():
    # Setup configuration
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 🌪️ Starting Test-Time Augmentation (TTA) Evaluation on {device} ===")

    # 1. Initialize Model
    net = ContUrbanCDModel(cfg)
    net = torch.nn.DataParallel(net)
    net.to(device)

    # Load SOTA weights
    checkpoint_path = args.resume 
    if not os.path.exists(checkpoint_path):
        print("❌ Error: Valid .pt file required.")
        sys.exit(1)

    print(f"📥 Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device).get('weights', torch.load(checkpoint_path))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    curr_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    matched_dict = {k: v for k, v in clean_dict.items() if k in curr_dict and v.shape == curr_dict[k].shape}
    
    target_model = net.module if hasattr(net, 'module') else net
    target_model.load_state_dict(matched_dict, strict=False)
    net.eval()

    # 2. Setup Data
    tile_size = cfg.AUGMENTATION.CROP_SIZE
    ds = datasets.create_eval_dataset(cfg, 'val', tiling=tile_size)
    # Using batch_size from config; shuffle must be False for evaluation
    dataloader = torch_data.DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=4, shuffle=False)
    edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.DATALOADER.EDGE_TYPE)
    
    m = evaluation.MultiTaskMeasurer()

    # 3. TTA Evaluation Loop
    with torch.no_grad():
        print("🚀 Running 4-Rotation TTA... (This will take 4x longer than normal validation)")
        for step, item in enumerate(dataloader):
            x = item['x'].to(device)
            y_seg = item['y'].to(device)
            y_ch = helpers.get_ch(y_seg, edges)
            
            # --- Pass 1: 0 degrees ---
            l_ch_0, _, r_seg_0 = net(x, edges)
            
            # --- Pass 2: 90 degrees ---
            x90 = torch.rot90(x, k=1, dims=[-2, -1])
            l_ch_90, _, r_seg_90 = net(x90, edges)
            # Rotate predictions back
            l_ch_90 = torch.rot90(l_ch_90, k=-1, dims=[-2, -1])
            r_seg_90 = torch.rot90(r_seg_90, k=-1, dims=[-2, -1])
            
            # --- Pass 3: 180 degrees ---
            x180 = torch.rot90(x, k=2, dims=[-2, -1])
            l_ch_180, _, r_seg_180 = net(x180, edges)
            # Rotate predictions back
            l_ch_180 = torch.rot90(l_ch_180, k=-2, dims=[-2, -1])
            r_seg_180 = torch.rot90(r_seg_180, k=-2, dims=[-2, -1])
            
            # --- Pass 4: 270 degrees ---
            x270 = torch.rot90(x, k=3, dims=[-2, -1])
            l_ch_270, _, r_seg_270 = net(x270, edges)
            # Rotate predictions back
            l_ch_270 = torch.rot90(l_ch_270, k=-3, dims=[-2, -1])
            r_seg_270 = torch.rot90(r_seg_270, k=-3, dims=[-2, -1])

            # Average the Sigmoid Probabilities
            o_ch = (torch.sigmoid(l_ch_0) + torch.sigmoid(l_ch_90) + torch.sigmoid(l_ch_180) + torch.sigmoid(l_ch_270)) / 4.0
            o_seg = (torch.sigmoid(r_seg_0) + torch.sigmoid(r_seg_90) + torch.sigmoid(r_seg_180) + torch.sigmoid(r_seg_270)) / 4.0

            m.add_sample(y_seg.cpu(), o_seg.cpu(), y_ch.cpu(), o_ch.cpu())

    # 4. Calculate Final SOTA Metrics
    f1_seg_cont = evaluation.f1_score(m.TP_seg_cont, m.FP_seg_cont, m.FN_seg_cont)
    f1_seg_fl = evaluation.f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl)
    f1_ch_cont = evaluation.f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
    f1_ch_fl = evaluation.f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
    
    f1_total = (f1_seg_cont + f1_seg_fl + f1_ch_cont + f1_ch_fl) / 4
    
    print("\n" + "="*40)
    print("🏆 FINAL TTA SOTA RESULTS 🏆")
    print("="*40)
    print(f"Total F1:        {f1_total:.4f}")
    print(f"Segmentation F1: {f1_seg_cont:.4f}")
    print(f"Change F1:       {f1_ch_cont:.4f}")
    print("="*40)

if __name__ == '__main__':
    evaluate_with_tta()