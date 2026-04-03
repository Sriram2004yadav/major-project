import sys
import os
import timeit
import torch
from torch import optim
from torch.utils import data as torch_data
import wandb
import numpy as np
from pathlib import Path

from utils import datasets, evaluation, experiment_manager, parsers, helpers
from model import model
from model.modules import TemporalConsistencyLoss
from utils.experiment_manager import CfgNode


def run_training(cfg: CfgNode):
    net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        lr_l = 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
        return lr_l

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    # Original loss — unchanged
    criterion = model.power_jaccard_loss

    # NEW: Temporal Consistency Loss
    tcl_criterion = TemporalConsistencyLoss(lambda_tcl=cfg.MODEL.LOSS.TCL_LAMBDA)

    dataset = datasets.create_train_dataset(cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size':  cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':     cfg.DATALOADER.SHUFFLE,
        'drop_last':   True,
        'pin_memory':  True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.MODEL.EDGE_TYPE)
    T     = cfg.DATALOADER.TIMESERIES_LENGTH

    epochs          = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # ── Resume from checkpoint if it exists ─────────────────────────
    best_f1_val = 0.0
    start_epoch = 1
    checkpoint_path = Path(cfg.PATHS.OUTPUT) / 'weights' / f'{cfg.NAME}.pt'

    if checkpoint_path.exists():
        print(f'=== Resuming from checkpoint: {checkpoint_path} ===')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['weights'])
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        best_f1_val = 0.515  # your confirmed best so far — won't overwrite with worse
        print(f'=== Loaded epoch {start_epoch - 1}. Best F1 so far: {best_f1_val} ===')
    else:
        print('=== No checkpoint found. Starting from scratch. ===')
    # ────────────────────────────────────────────────────────────────

    trigger = 0
    stop    = False

    global_step = (start_epoch - 1) * steps_per_epoch
    epoch_float = float(start_epoch - 1)

    for epoch in range(start_epoch, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({
            'lr':    scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR,
            'epoch': epoch,
        })

        start = timeit.default_timer()
        loss_seg_set, loss_cal_set, loss_ch_set, loss_tcl_set, loss_set = [], [], [], [], []

        for i, batch in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()

            x, y_seg = batch['x'].to(device), batch['y'].to(device)

            logits_ch, logits_seg, calibrated_logits = net(x, edges)

            y_ch = helpers.get_ch(y_seg, edges)

            loss_seg = criterion(logits_seg, y_seg)
            loss_ch  = criterion(logits_ch,  y_ch)
            loss_cal = criterion(calibrated_logits, y_seg)
            loss_tcl = tcl_criterion(logits_ch, edges, T)

            loss = loss_seg + loss_cal + loss_ch + loss_tcl

            loss.backward()
            optimizer.step()

            loss_seg_set.append(loss_seg.item())
            loss_cal_set.append(loss_cal.item())
            loss_ch_set.append(loss_ch.item())
            loss_tcl_set.append(loss_tcl.item())
            loss_set.append(loss.item())

            global_step += 1
            epoch_float  = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                time = timeit.default_timer() - start
                wandb.log({
                    'loss_seg': np.mean(loss_seg_set),
                    'loss_cal': np.mean(loss_cal_set),
                    'loss_ch':  np.mean(loss_ch_set),
                    'loss_tcl': np.mean(loss_tcl_set),
                    'loss':     np.mean(loss_set),
                    'time':     time,
                    'step':     global_step,
                    'epoch':    epoch_float,
                })
                start = timeit.default_timer()
                loss_seg_set, loss_cal_set, loss_ch_set, loss_tcl_set, loss_set = [], [], [], [], []

        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')

        if scheduler is not None:
            scheduler.step()

        # ── Evaluate every 10 epochs only ───────────────────────────
        if epoch % 10 == 0:
            print(f'Running validation at epoch {epoch}...')
            f1_val = evaluation.model_evaluation(
                net, cfg, device, 'val', epoch_float, global_step
            )

            if f1_val <= best_f1_val:
                trigger += 1
                print(f'No improvement. Trigger {trigger}/{cfg.TRAINER.PATIENCE}. Best: {best_f1_val:.4f}')
                if trigger > cfg.TRAINER.PATIENCE:
                    stop = True
            else:
                best_f1_val = f1_val
                trigger     = 0
                wandb.log({
                    'best val f1': best_f1_val,
                    'step':        global_step,
                    'epoch':       epoch_float,
                })
                print(f'saving network (F1 {f1_val:.3f})', flush=True)
                model.save_model(net, epoch, cfg)
        else:
            print(f'Skipping validation (next at epoch {((epoch // 10) + 1) * 10}).')
        # ────────────────────────────────────────────────────────────

        if stop:
            print('Early stopping triggered.')
            break

    # Final test evaluation on best saved weights
    print('Loading best model for final test evaluation...')
    net = model.load_model(cfg, device)
    _   = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg  = experiment_manager.setup_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Running on device:', device)

    wandb.init(
        name    = cfg.NAME,
        config  = cfg,
        project = 'ContUrbanCD',
        mode    = 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)