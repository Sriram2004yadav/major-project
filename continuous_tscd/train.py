import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import datasets, evaluation, experiment_manager, parsers, helpers
from model import model
from utils.experiment_manager import CfgNode


def run_training(cfg: CfgNode):
    net = model.init_model(cfg)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    def lambda_rule(e: int):
        lr_l = 1.0 - e / float(cfg.TRAINER.EPOCHS - 1)
        return lr_l
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    criterion = model.power_jaccard_loss

    # reset the generators
    dataset = datasets.create_train_dataset(cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, cfg.MODEL.EDGE_TYPE)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val = 0
    trigger_times = 0
    stop_training = False

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')
        wandb.log({'lr': scheduler.get_last_lr()[-1] if scheduler is not None else cfg.TRAINER.LR, 'epoch': epoch})
        start = timeit.default_timer()
        loss_seg_set, loss_ch_set, loss_set = [], [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            logits_ch, logits_seg = net(x, edges)

            # --- START OF MODIFICATION ---
            # Conditional logic to handle different datasets.
            # TSCD dataset only provides change labels ('y_ch').
            # SN7/WUSU provide segmentation labels ('y'), and change is derived.
            
            if cfg.DATASET.NAME == 'tscd':
                # For TSCD, we load 'y_ch' directly from the dataset
                y_ch = batch['y_ch'].to(device)
                
                # As per Table V in the paper, Seg loss is OFF for TSCD
                loss_ch = criterion(logits_ch, y_ch)
                loss_seg = 0.0 # No segmentation loss
                loss = loss_ch
            
            else:
                # This is the original logic for SN7 and WUSU
                y_seg = batch['y'].to(device)
                y_ch = helpers.get_ch(y_seg, edges)
    
                loss_seg = criterion(logits_seg, y_seg)
                loss_ch = criterion(logits_ch, y_ch)
    
                loss = loss_seg + loss_ch
            
            # --- END OF MODIFICATION ---

            loss.backward()
            optimizer.step()

            # --- START OF MODIFICATION ---
            # Also modify the logging to avoid errors
            
            if loss_seg != 0.0:
                loss_seg_set.append(loss_seg.item())
            loss_ch_set.append(loss_ch.item())
            loss_set.append(loss.item())
            # --- END OF MODIFICATION ---

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # logging
                time = timeit.default_timer() - start
                
                # --- START OF MODIFICATION ---
                # Make logging conditional
                log_data = {
                    'loss_ch': np.mean(loss_ch_set),
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                }
                if len(loss_seg_set) > 0:
                    log_data['loss_seg'] = np.mean(loss_seg_set)
                
                wandb.log(log_data)
                # --- END OF MODIFICATION ---
                
                start = timeit.default_timer()
                loss_seg_set, loss_ch_set, loss_set = [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        if scheduler is not None:
            scheduler.step()
        # evaluation at the end of an epoch
        f1_val = evaluation.model_evaluation(net, cfg, device, 'val', epoch_float, global_step)

        if f1_val <= best_f1_val:
            trigger_times += 1
            if trigger_times > cfg.TRAINER.PATIENCE:
                stop_training = True
        else:
            best_f1_val = f1_val
            wandb.log({
                'best val f1': best_f1_val,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            model.save_model(net, epoch, cfg)
            trigger_times = 0

        if stop_training:
            break

    net = model.load_model(cfg, device)
    _ = evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        project='ContUrbanCD',
        mode='disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)