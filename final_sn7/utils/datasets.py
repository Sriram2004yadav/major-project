import torch
from torch.utils import data as torch_data
import numpy as np
from pathlib import Path
import rasterio
from rasterio.windows import Window
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import os

def create_train_dataset(cfg, run_type='train'):
    return TrainSpaceNet7Dataset(cfg, run_type)

def create_eval_dataset(cfg, run_type, site=None, tiling=None):
    return EvalSpaceNet7Dataset(cfg, run_type=run_type, site=site, tiling=tiling)

class TrainSpaceNet7Dataset(torch_data.Dataset):
    def __init__(self, cfg, run_type='train'):
        self.cfg = cfg
        self.root = Path(cfg.PATHS.DATASET if hasattr(cfg.PATHS, 'DATASET') else cfg.DATASET.ROOT)
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE
        self.T = cfg.DATALOADER.TIMESERIES_LENGTH
        
        with open('metadata_conturbancd.json', 'r') as f:
            raw_metadata = json.load(f)

        self.aoi_ids = cfg.DATASET.TRAIN_IDS if run_type == 'train' else cfg.DATASET.VAL_IDS
        self.clean_metadata = {}
        valid_aois = []

        print(f"Pre-scanning {run_type} directories...")
        for aoi in self.aoi_ids:
            img_dir = self.root / 'train' / aoi / 'images_masked'
            lbl_dir = self.root / 'train' / aoi / 'labels_raster' / 'labels_raster'
            
            aoi_files = []
            if aoi in raw_metadata:
                for ts in raw_metadata[aoi]:
                    y, m = ts['year'], int(ts['month'])
                    # Logic: use zero-padding (01, 02) to match your file system
                    img_n = f"global_monthly_{y}_{m:02d}_mosaic_{aoi}.tif"
                    lbl_n = f"global_monthly_{y}_{m:02d}_mosaic_{aoi}_Buildings.tif"
                    
                    if (img_dir / img_n).exists() and (lbl_dir / lbl_n).exists():
                        aoi_files.append({'img': img_dir / img_n, 'lbl': lbl_dir / lbl_n})
            
            if len(aoi_files) >= 1:
                self.clean_metadata[aoi] = aoi_files
                valid_aois.append(aoi)

        self.aoi_ids = valid_aois
        self.multiplier = cfg.DATALOADER.TRAINING_MULTIPLIER
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), ToTensorV2()
        ])

    def __len__(self):
        return len(self.aoi_ids) * self.multiplier

    def __getitem__(self, index):
        aoi_id = self.aoi_ids[index % len(self.aoi_ids)]
        files = self.clean_metadata[aoi_id]
        selected = files[random.randint(0, len(files)-self.T):][:self.T] if len(files) >= self.T else (files * self.T)[:self.T]

        # Dynamic shape detection (fixes 1023 vs 1024 errors)
        with rasterio.open(selected[0]['img']) as src:
            h_max, w_max = src.height, src.width
        
        window = Window(random.randint(0, w_max - self.crop_size), 
                        random.randint(0, h_max - self.crop_size), 
                        self.crop_size, self.crop_size)

        imgs, lbls = [], []
        for f in selected:
            with rasterio.open(f['img']) as src:
                imgs.append(src.read(window=window)[:3].transpose(1, 2, 0))
            with rasterio.open(f['lbl']) as src:
                lbls.append(src.read(1, window=window))

        out = self.transform(image=np.concatenate(imgs, axis=2), mask=np.stack(lbls, axis=2))
        x = out['image'].float() / 255.0
        x = x.view(self.T, 3, self.crop_size, self.crop_size)
        y = out['mask'].permute(2, 0, 1).float().unsqueeze(1)
        return {'x': x, 'y': (y > 0).float()}

class EvalSpaceNet7Dataset(torch_data.Dataset):
    def __init__(self, cfg, run_type='val', site=None, tiling=None):
        self.cfg = cfg
        self.root = Path(cfg.PATHS.DATASET if hasattr(cfg.PATHS, 'DATASET') else cfg.DATASET.ROOT)
        self.T, self.tile_size = cfg.DATALOADER.TIMESERIES_LENGTH, (tiling or 256)
        with open('metadata_conturbancd.json', 'r') as f:
            raw_metadata = json.load(f)

        self.aoi_ids = [site] if site else cfg.DATASET.VAL_IDS
        self.clean_metadata, self.tiles = {}, []

        for aoi in self.aoi_ids:
            img_dir = self.root / 'train' / aoi / 'images_masked'
            lbl_dir = self.root / 'train' / aoi / 'labels_raster' / 'labels_raster'
            valid = []
            if aoi in raw_metadata:
                for ts in raw_metadata[aoi]:
                    y, m = ts['year'], int(ts['month'])
                    path = img_dir / f"global_monthly_{y}_{m:02d}_mosaic_{aoi}.tif"
                    if path.exists():
                        valid.append({'img': path, 'lbl': lbl_dir / f"global_monthly_{y}_{m:02d}_mosaic_{aoi}_Buildings.tif"})
            if valid:
                self.clean_metadata[aoi] = valid
                with rasterio.open(valid[0]['img']) as src:
                    h, w = src.height, src.width
                for i in range(0, h - self.tile_size + 1, self.tile_size):
                    for j in range(0, w - self.tile_size + 1, self.tile_size):
                        self.tiles.append({'aoi': aoi, 'i': i, 'j': j})

    def __len__(self): return len(self.tiles)

    def __getitem__(self, index):
        t = self.tiles[index]
        aoi, i, j = t['aoi'], t['i'], t['j']
        files = (self.clean_metadata[aoi][:self.T] if len(self.clean_metadata[aoi]) >= self.T else (self.clean_metadata[aoi] * self.T)[:self.T])
        window = Window(j, i, self.tile_size, self.tile_size)
        imgs, lbls = [], []
        for f in files:
            with rasterio.open(f['img']) as src: imgs.append(src.read(window=window)[:3])
            with rasterio.open(f['lbl']) as src: lbls.append(src.read(1, window=window))
        return {'x': torch.from_numpy(np.stack(imgs)).float() / 255.0, 'y': torch.from_numpy(np.stack(lbls)).float().unsqueeze(1), 'i': i, 'j': j, 'site': aoi}