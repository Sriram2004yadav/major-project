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

# --- FACTORY FUNCTIONS ---
def create_train_dataset(cfg, run_type='train'):
    return TrainWUSUDataset(cfg, run_type)

def create_eval_dataset(cfg, run_type, site=None, tiling=None):
    if run_type == 'test':
        return EvalTestWUSUDataset(cfg, site=site, tiling=tiling)
    if tiling is None:
        tiling = cfg.AUGMENTATION.CROP_SIZE
    return EvalWUSUDataset(cfg, run_type=run_type, site=site, tiling=tiling)


# --- OPTIMIZED DATASET CLASSES ---

class TrainWUSUDataset(torch_data.Dataset):
    def __init__(self, cfg, run_type='train'):
        self.cfg = cfg
        self.root = Path(cfg.DATASET.ROOT)
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE
        self.timestamps = [15, 16, 18]
        
        with open(self.root / 'samples_train.json', 'r') as f:
            all_samples = json.load(f)
        self.samples = [s for s in all_samples if s['split'] == run_type]
        
        # Cache image dimensions to avoid opening files repeatedly to check size
        self.metadata_cache = {} 
        
        # --- OPTIMIZATION: Removed RandomCrop from here ---
        # We manually crop using Window reading for speed.
        # Only keep spatial augmentations.
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

    def __len__(self):
        # --- OPTIMIZATION: Reduced multiplier from 100 to 25 ---
        # This gives you 4x more frequent validation checks within your 6 hours.
        return len(self.samples) * 25

    def _get_metadata(self, img_path):
        path_str = str(img_path)
        if path_str not in self.metadata_cache:
            with rasterio.open(img_path) as src:
                self.metadata_cache[path_str] = (src.height, src.width)
        return self.metadata_cache[path_str]

    def __getitem__(self, index):
        sample_idx = index % len(self.samples)
        sample_info = self.samples[sample_idx]
        site = sample_info['site']
        idx_str = sample_info['index']
        
        base_dir = self.root / 'train' / site
        img_dir = base_dir / 'imgs'
        label_dir = base_dir / 'class'
        
        # 1. Determine Crop Coordinates (Randomly)
        # Check dimensions of the first image (assuming all timestamps match)
        ref_img = img_dir / f"{site}15_{idx_str}.tif"
        H, W = self._get_metadata(ref_img)
        
        # Randomly pick top-left corner
        h_start = random.randint(0, H - self.crop_size)
        w_start = random.randint(0, W - self.crop_size)
        
        # Create the Window object (This is the IO Speed Hack)
        # We only read this tiny square!
        window = Window(w_start, h_start, self.crop_size, self.crop_size)
        
        imgs = []
        labels = []
        
        for year in self.timestamps:
            filename = f"{site}{year}_{idx_str}.tif"
            
            # Read Image Window
            with rasterio.open(img_dir / filename) as src:
                img = src.read(window=window).transpose(1, 2, 0) # H,W,C
                imgs.append(img)
            
            # Read Label Window
            with rasterio.open(label_dir / filename) as src:
                lbl = src.read(1, window=window) # H,W
                lbl = (lbl == 2) | (lbl == 3)
                labels.append(lbl.astype(np.float32))

        stacked_imgs = np.concatenate(imgs, axis=2)
        stacked_labels = np.stack(labels, axis=2)
        
        # Apply Transforms (Flips/Rotates only)
        transformed = self.transform(image=stacked_imgs, mask=stacked_labels)
        
        img_tensor = transformed['image']
        lbl_tensor = transformed['mask']
        
        T = 3
        C = img_tensor.shape[0] // T
        img_tensor = img_tensor.view(T, C, self.crop_size, self.crop_size)
        lbl_tensor = lbl_tensor.permute(2, 0, 1).float().unsqueeze(1)
        
        return {'x': img_tensor.float(), 'y': lbl_tensor}

# --- Keep Eval classes same as before ---
class EvalWUSUDataset(torch_data.Dataset):
    def __init__(self, cfg, run_type='test', site=None, tiling=256):
        self.cfg = cfg
        self.root = Path(cfg.DATASET.ROOT)
        self.tile_size = tiling
        self.timestamps = [15, 16, 18]
        
        json_name = 'samples_train.json'
        self.folder = 'train'
        with open(self.root / json_name, 'r') as f:
            all_samples = json.load(f)
        if run_type == 'val':
            self.samples = [s for s in all_samples if s['split'] == 'val']
        else:
            self.samples = all_samples
        if site:
            self.samples = [s for s in self.samples if s['site'] == site]

        self.tiles = []
        for s in self.samples:
            site_name = s['site']
            idx_str = s['index']
            base_dir = self.root / self.folder / site_name
            filename = f"{site_name}15_{idx_str}.tif"
            ref_img = base_dir / 'imgs' / filename
            if not ref_img.exists(): continue
            with rasterio.open(ref_img) as src:
                H, W = src.height, src.width
            for i in range(0, H, self.tile_size):
                for j in range(0, W, self.tile_size):
                    if i + self.tile_size <= H and j + self.tile_size <= W:
                        self.tiles.append({'site': site_name, 'index': idx_str, 'i': i, 'j': j, 'folder': self.folder})

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        site = tile['site']
        idx_str = tile['index']
        folder = tile['folder']
        i, j = tile['i'], tile['j']
        
        base_dir = self.root / folder / site
        imgs = []
        labels = []
        for year in self.timestamps:
            filename = f"{site}{year}_{idx_str}.tif"
            with rasterio.open(base_dir / 'imgs' / filename) as src:
                window = Window(j, i, self.tile_size, self.tile_size)
                img = src.read(window=window)
                imgs.append(torch.from_numpy(img).float())
            with rasterio.open(base_dir / 'class' / filename) as src:
                window = Window(j, i, self.tile_size, self.tile_size)
                lbl = src.read(1, window=window)
                lbl = (lbl == 2) | (lbl == 3)
                labels.append(torch.from_numpy(lbl.astype(np.float32)).float())
        
        img_tensor = torch.stack(imgs, dim=0)
        lbl_tensor = torch.stack(labels, dim=0).unsqueeze(1)
        return {'x': img_tensor, 'y': lbl_tensor, 'i': i, 'j': j}

class EvalTestWUSUDataset(torch_data.Dataset):
    def __init__(self, cfg, site=None, tiling=None):
        self.cfg = cfg
        self.root = Path(cfg.DATASET.ROOT)
        self.tile_size = tiling if tiling is not None else cfg.AUGMENTATION.CROP_SIZE
        self.timestamps = [15, 16, 18]
        with open(self.root / 'samples_test.json', 'r') as f:
            all_samples = json.load(f)
        if site:
            self.samples = [s for s in all_samples if s['site'] == site]
        else:
            self.samples = all_samples
        self.tiles = []
        for s in self.samples:
            site_name = s['site']
            idx_str = s['index']
            base_dir = self.root / 'test' / site_name
            filename = f"{site_name}15_{idx_str}.tif"
            ref_img = base_dir / 'imgs' / filename
            if not ref_img.exists():
                possible = list((base_dir / 'imgs').glob("*.tif"))
                if possible: ref_img = possible[0]
                else: continue
            with rasterio.open(ref_img) as src:
                H, W = src.height, src.width
            for i in range(0, H, self.tile_size):
                for j in range(0, W, self.tile_size):
                    if i + self.tile_size <= H and j + self.tile_size <= W:
                        self.tiles.append({'site': site_name, 'index': idx_str, 'i': i, 'j': j, 'folder': 'test'})

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        site = tile['site']
        idx_str = tile['index']
        folder = tile['folder']
        i, j = tile['i'], tile['j']
        base_dir = self.root / folder / site
        imgs = []
        labels = []
        for year in self.timestamps:
            filename = f"{site}{year}_{idx_str}.tif"
            with rasterio.open(base_dir / 'imgs' / filename) as src:
                window = Window(j, i, self.tile_size, self.tile_size)
                img = src.read(window=window)
                imgs.append(torch.from_numpy(img).float())
            with rasterio.open(base_dir / 'class' / filename) as src:
                window = Window(j, i, self.tile_size, self.tile_size)
                lbl = src.read(1, window=window)
                lbl = (lbl == 2) | (lbl == 3)
                labels.append(torch.from_numpy(lbl.astype(np.float32)).float())
        img_tensor = torch.stack(imgs, dim=0)
        lbl_tensor = torch.stack(labels, dim=0).unsqueeze(1)
        return {'x': img_tensor, 'y': lbl_tensor, 'i': i, 'j': j, 'site': site}