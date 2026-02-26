import sys
import torch
from torch import optim
from torch.utils import data as torch_data
from torch.amp import GradScaler, autocast 
import numpy as np
from collections import OrderedDict

from utils import datasets, evaluation, experiment_manager, parsers, helpers
from model.model import ContUrbanCDModel, save_model
from engine.losses import HybridSN7Loss

def run_training(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model
    net = ContUrbanCDModel(cfg)
    net = torch.nn.DataParallel(net).to(device)

    # 2. Universal Surgery Loader
    if args.resume:
        print(f"=== 🏥 Performing Surgery from: {args.resume} ===")
        state_dict = torch.load(args.resume, map_location=device).get('weights', torch.load(args.resume))
        clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
        curr_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        
        matched_dict = {k: v for k, v in clean_dict.items() if k in curr_dict and v.shape == curr_dict[k].shape}
        
        target_model = net.module if hasattr(net, 'module') else net
        target_model.load_state_dict(matched_dict, strict=False)
        
        print(f"✅ Transplanted {len(matched_dict)} layers. Skipped {len(curr_dict) - len(matched_dict)} mismatches.")

    # 3. Hybrid Losses Setup
    bce_w = cfg.MODEL.LOSS.BCE_WEIGHT
    jac_w = cfg.MODEL.LOSS.JACCARD_WEIGHT
    crit_seg = HybridSN7Loss(pos_weight=1.5, bce_weight=bce_w, jaccard_weight=jac_w).to(device)
    crit_ref = HybridSN7Loss(pos_weight=2.5, bce_weight=bce_w, jaccard_weight=jac_w).to(device)
    crit_ch = HybridSN7Loss(pos_weight=2.0, bce_weight=bce_w, jaccard_weight=jac_w).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAINER.EPOCHS)
    scaler = GradScaler('cuda') 

    dataset = datasets.create_train_dataset(cfg, run_type='train')
    dataloader = torch_data.DataLoader(dataset, batch_size=cfg.TRAINER.BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
    edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.DATALOADER.EDGE_TYPE)

    steps_per_epoch = len(dataloader)
    global_step = 0
    best_f1_val = 0.0

    # Training Loop
    for epoch in range(1, cfg.TRAINER.EPOCHS + 1):
        print(f'Starting epoch {epoch}.')
        loss_set = []

        for batch in dataloader:
            net.train()
            optimizer.zero_grad()
            x = batch['x'].to(device) 
            y_seg = batch['y'].to(device)
            y_ch = helpers.get_ch(y_seg, edges).to(device)

            with autocast('cuda'):
                logits_ch, logits_seg, refined_seg = net(x, edges)
                
                l_seg = crit_seg(logits_seg, y_seg)
                l_ref = crit_ref(refined_seg, y_seg)
                l_ch = crit_ch(logits_ch, y_ch)
                loss = l_seg + l_ref + l_ch

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_set.append(loss.item())
            global_step += 1

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Step {global_step} - Loss: {np.mean(loss_set):.4f}')
                loss_set = []

        if scheduler: scheduler.step()
        
        print("Running Validation...")
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch, global_step)

        if f1_val > best_f1_val:
            best_f1_val = f1_val
            print(f'New Best F1: {f1_val:.3f}. Saving...')
            save_model(net, epoch, cfg)

if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    torch.backends.cudnn.benchmark = True 
    run_training(cfg, args)