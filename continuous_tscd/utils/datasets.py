import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
import tifffile
from utils import experiment_manager, helpers
from utils.experiment_manager import CfgNode
import cv2
import torch
from torchvision import transforms
import numpy as np
from scipy.ndimage import gaussian_filter


def create_train_dataset(cfg: CfgNode, run_type: str) -> torch.utils.data.Dataset:
    if cfg.DATASET.NAME == 'sn7':
        return TrainSpaceNet7Dataset(cfg, run_type=run_type)
    elif cfg.DATASET.NAME == 'wusu':
        return TrainWUSUDataset(cfg, run_type=run_type)
    elif cfg.DATASET.NAME == 'tscd':  # <--- ADD THIS LINE
        return TrainTSCDDataset(cfg, run_type=run_type)
    else:
        raise Exception('Unknown train dataset!')



def create_eval_dataset(cfg: CfgNode, run_type: str, site: str = None, tiling: int = None) -> torch.utils.data.Dataset:
    if cfg.DATASET.NAME == 'sn7':
        return EvalSpaceNet7Dataset(cfg, run_type, aoi_id=site, tiling=tiling)
    elif cfg.DATASET.NAME == 'wusu':
        if run_type == 'test':
            return EvalTestWUSUDataset(cfg, site=site)
        else:
            return EvalWUSUDataset(cfg, run_type=run_type, tiling=tiling)
    # --- START MODIFICATION ---
    elif cfg.DATASET.NAME == 'tscd': 
        if run_type == 'test':
            # We will add this class in the next step
            return EvalTestTSCDDataset(cfg, site=site) 
        else:
            # This will fix your current error
            return EvalTSCDDataset(cfg, run_type=run_type, tiling=tiling) 
    # --- END MODIFICATION ---
    else:
        raise Exception('Unknown train dataset!')

class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.name = cfg.DATASET.NAME

        self.include_alpha = cfg.DATALOADER.INCLUDE_ALPHA
        self.pad = cfg.DATALOADER.PAD_BORDERS

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def load_planet_mosaic(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        img = tifffile.imread(str(file))
        img = img / 255
        # 4th band (last oen) is alpha band
        if not self.include_alpha:
            img = img[:, :, :-1]
        m, n, _ = img.shape
        if self.pad and (m != 1024 or n != 1024):
            # https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/
            img = cv2.copyMakeBorder(img, 0, 1024 - m, 0, 1024 - n, borderType=cv2.BORDER_REPLICATE)
        return img.astype(np.float32)

    def load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label = tifffile.imread(str(file))
        m, n = label.shape
        if self.pad and (m != 1024 or n != 1024):
            label = cv2.copyMakeBorder(label, 0, 1024 - m, 0, 1024 - n, borderType=cv2.BORDER_REPLICATE)
        label = label[:, :, None]
        label = label > 0
        return label.astype(np.float32)

    def load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self.load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self.load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def load_mask(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_mask.tif'
        mask = tifffile.imread(str(file))
        return mask.astype(np.int8)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class AbstractWUSUDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.timestamps = [15, 16, 18]
        self.T = 3
        self.name = cfg.DATASET.NAME

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def load_gf2_img(self, site: str, dataset: str, index: int, year: int) -> np.ndarray:
        file = self.root_path / dataset / site / 'imgs' / f'{site}{year}_{index}.tif'
        img = tifffile.imread(str(file))
        img = img / 255
        return img.astype(np.float32)

    def load_lulc_label(self, site: str, dataset: str, index: int, year: int) -> np.ndarray:
        file = self.root_path / dataset / site / 'class' / f'{site}{year}_{index}.tif'
        lulc_label = tifffile.imread(str(file))
        lulc_label = lulc_label[:, :, None]
        return lulc_label

    def load_building_label(self, site: str, dataset: str, index: int, year: int) -> np.ndarray:
        lulc = self.load_lulc_label(site, dataset, index, year)
        buildings = np.logical_or(lulc == 2, lulc == 3)
        return buildings.astype(np.float32)

    def load_building_change_label(self, site: str, dataset: str, index: int, year_t1: int, year_t2: int) -> np.ndarray:
        buildings_t1 = self.load_building_label(site, dataset, index, year_t1)
        buildings_t2 = self.load_building_label(site, dataset, index, year_t2)
        change = np.logical_and(buildings_t1 == 0, buildings_t2 == 1)
        return change.astype(np.float32)


class TrainWUSUDataset(AbstractWUSUDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = compose_transformations(cfg, no_augmentations)

        self.dataset = 'test' if run_type == 'test' else 'train'
        if run_type == 'test':
            self.samples = helpers.load_json(self.root_path / f'samples_test.json')
        elif run_type == 'val' or run_type == 'train':
            self.samples = helpers.load_json(self.root_path / f'samples_train.json')
            self.samples = [s for s in self.samples if s['split'] == run_type]
        else:
            raise Exception('Unkown run type!')

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        site, index = sample['site'], sample['index']

        images = [self.load_gf2_img(site, self.dataset, index, year) for year in self.timestamps]
        labels = [self.load_building_label(site, self.dataset, index, year) for year in self.timestamps]

        images, labels = self.transform((np.stack(images), np.stack(labels)))

        item = {
            'x': images,
            'y': labels,
            'site': site,
            'index': index,
        }
        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Train {self.name} dataset with {self.length} samples.'


class EvalWUSUDataset(AbstractWUSUDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, tiling: int = None, add_padding: bool = False,
                 index: int = None):
        super().__init__(cfg)

        self.tiling = tiling
        self.add_padding = add_padding

        # handling transformations of data
        self.transform = compose_transformations(cfg, no_augmentations=True)

        self.dataset = 'test' if run_type == 'test' else 'train'
        if run_type == 'test':
            samples = helpers.load_json(self.root_path / f'samples_test.json')
        elif run_type == 'val' or run_type == 'train':
            samples = helpers.load_json(self.root_path / f'samples_train.json')
            samples = [s for s in samples if s['split'] == run_type]
        else:
            raise Exception('Unkown run type!')

        if index is not None:
            samples = [s for s in samples if s['index'] == index]

        if tiling is None:
            self.tiling = 512

        self.samples = []
        for sample in samples:
            for i in range(0, 512, self.tiling):
                for j in range(0, 512, self.tiling):
                    tile_sample = {
                        'site': sample['site'],
                        'index': sample['index'],
                        'split': sample['split'],
                        'i': i,
                        'j': j,
                    }
                    self.samples.append(tile_sample)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        site, index, i, j = sample['site'], sample['index'], sample['i'], sample['j']

        images = [self.load_gf2_img(site, self.dataset, index, year) for year in self.timestamps]
        images = np.stack(images)
        if self.add_padding:
            images = np.pad(images, ((0, 0), (self.tiling, self.tiling), (self.tiling, self.tiling), (0, 0)),
                            mode='reflect')
            i_min, j_min = i, j
            i_max, j_max = i + 3 * self.tiling, j + 3 * self.tiling
            images = images[:, i_min:i_max, j_min:j_max]
        else:
            images = images[:, i:i + self.tiling, j:j + self.tiling]

        labels = [self.load_building_label(site, self.dataset, index, year) for year in self.timestamps]
        labels = np.stack(labels)[:, i:i + self.tiling, j:j + self.tiling]

        images, labels = self.transform((images, labels))

        item = {
            'x': images,
            'y': labels,
            'site': site,
            'index': index,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Eval {self.name} dataset with {self.length} samples.'


class EvalTestWUSUDataset(AbstractWUSUDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, site: str = None):
        super().__init__(cfg)

        metadata_file = self.root_path / 'metadata_test.json'
        self.metadata = helpers.load_json(metadata_file)

        self.sites = ['JA', 'HS'] if site is None else [site]

        self.transform = compose_transformations(cfg, no_augmentations=True)
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE

        self.dataset = 'test'
        self.samples = []
        for site in self.sites:
            tiles = self.metadata[site]['tiles']
            tile_size = int(self.metadata[site]['tile_size'] - self.metadata[site]['overlap'])
            for tile in tiles:
                i_tile, j_tile = tile['i_tile'], tile['j_tile']
                m_max, n_max = tile_size, tile_size
                if tile['edge_tile']:
                    m_edge_tile, n_edge_tile = self.metadata[site]['m_edge_tile'], self.metadata[site]['n_edge_tile']
                    m_edge_tile = m_edge_tile - m_edge_tile % self.crop_size
                    n_edge_tile = n_edge_tile - n_edge_tile % self.crop_size
                    if tile['row_end']:
                        m_max = m_edge_tile
                    if tile['col_end']:
                        n_max = n_edge_tile

                for i in range(0, m_max, self.crop_size):
                    for j in range(0, n_max, self.crop_size):
                        tile_sample = {
                            'site': site,
                            'index': tile['index'],
                            'split': 'test',
                            'i_crop': i,
                            'j_crop': j,
                            'i_tile': i_tile,
                            'j_tile': j_tile,
                        }
                        self.samples.append(tile_sample)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.metadata = manager.dict(self.metadata)
        self.length = len(self.samples)

    def __getitem__(self, sample_index):

        sample = self.samples[sample_index]
        site, index = sample['site'], sample['index']

        images = np.stack([self.load_gf2_img(site, self.dataset, index, year) for year in self.timestamps])
        i_crop, j_crop = sample['i_crop'], sample['j_crop']
        images = images[:, i_crop:i_crop + self.crop_size, j_crop:j_crop + self.crop_size]

        labels = np.stack([self.load_building_label(site, self.dataset, index, year) for year in self.timestamps])
        labels = labels[:, i_crop:i_crop + self.crop_size, j_crop:j_crop + self.crop_size]

        images, labels = self.transform((images, labels))

        i_tile, j_tile = sample['i_tile'], sample['j_tile']
        tile_size = int(self.metadata[site]['tile_size'] - self.metadata[site]['overlap'])
        i_img, j_img = i_tile * tile_size + i_crop, j_tile * tile_size + j_crop

        item = {
            'x': images,
            'y': labels,
            'site': site,
            'index': index,
            'i_img': i_img,
            'j_img': j_img,
        }

        return item

    def get_img_dims(self, site: str) -> tuple:
        tile_size = int(self.metadata[site]['tile_size'] - self.metadata[site]['overlap'])
        m_tile, n_tile = self.metadata[site]['m_tile'], self.metadata[site]['n_tile']

        m_edge_tile, n_edge_tile = self.metadata[site]['m_edge_tile'], self.metadata[site]['n_edge_tile']
        m_edge_tile = m_edge_tile - m_edge_tile % self.crop_size
        n_edge_tile = n_edge_tile - n_edge_tile % self.crop_size

        m_img = (m_tile - 1) * tile_size + m_edge_tile
        n_img = (n_tile - 1) * tile_size + n_edge_tile
        return m_img, n_img

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Eval {self.name} dataset with {self.length} samples.'


class TrainSpaceNet7Dataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False,
                 disable_multiplier: bool = False):
        super().__init__(cfg)

        self.T = cfg.DATALOADER.TIMESERIES_LENGTH
        self.include_change_label = cfg.DATALOADER.INCLUDE_CHANGE_LABEL

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = compose_transformations(cfg, no_augmentations)

        self.metadata = helpers.load_json(self.root_path / f'metadata_siamesessl.json')
        self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
        assert (len(self.aoi_ids) == 60)

        # split
        self.run_type = run_type
        self.i_split = cfg.DATALOADER.I_SPLIT
        self.j_split = cfg.DATALOADER.J_SPLIT

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]

        timestamps = [ts for ts in self.metadata[aoi_id] if not ts['mask']]

        t_values = sorted(np.random.randint(0, len(timestamps), size=self.T))
        timestamps = sorted([timestamps[t] for t in t_values], key=lambda ts: int(ts['year']) * 12 + int(ts['month']))

        images = [self.load_planet_mosaic(aoi_id, ts['dataset'], ts['year'], ts['month']) for ts in timestamps]
        labels = [self.load_building_label(aoi_id, ts['year'], ts['month']) for ts in timestamps]
        images = [self.apply_split(img) for img in images]
        labels = [self.apply_split(label) for label in labels]

        images, labels = self.transform((np.stack(images), np.stack(labels)))

        item = {
            'x': images,
            'y': labels,
            'aoi_id': aoi_id,
            'dates': [(int(ts['year']), int(ts['month'])) for ts in timestamps],
        }

        if self.include_change_label:
            labels_ch = []
            for t in range(len(timestamps) - 1):
                labels_ch.append(torch.ne(labels[t + 1], labels[t]))
            labels_ch.append(torch.ne(labels[-1], labels[0]))
            item['y_ch'] = torch.stack(labels_ch)

        return item

    def apply_split(self, img: np.ndarray):
        if self.run_type == 'train':
            return img[:self.i_split]
        elif self.run_type == 'val':
            return img[self.i_split:, :self.j_split]
        elif self.run_type == 'test':
            return img[self.i_split:, self.j_split:]
        else:
            raise Exception('Unkown split!')

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Train {self.name} dataset with {self.length} samples.'


class EvalSpaceNet7Dataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, tiling: int = None, aoi_id: str = None,
                 add_padding: bool = False):
        super().__init__(cfg)

        self.T = cfg.DATALOADER.TIMESERIES_LENGTH
        self.include_change_label = cfg.DATALOADER.INCLUDE_CHANGE_LABEL
        self.tiling = tiling if tiling is not None else 1024
        self.eval_train_threshold = cfg.DATALOADER.EVAL_TRAIN_THRESHOLD
        self.add_padding = add_padding

        # handling transformations of data
        self.transform = compose_transformations(cfg, no_augmentations=True)

        self.metadata = helpers.load_json(self.root_path / f'metadata_conturbancd.json')

        if aoi_id is None:
            self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
            assert (len(self.aoi_ids) == 60)
        else:
            self.aoi_ids = [aoi_id]

        # split
        self.run_type = run_type
        self.i_split = cfg.DATALOADER.I_SPLIT
        self.j_split = cfg.DATALOADER.J_SPLIT

        self.min_m, self.max_m = 0, 1024
        self.min_n, self.max_n = 0, 1024
        if run_type == 'train':
            self.max_m = self.i_split
            self.m = self.max_m
        else:
            assert (run_type == 'val' or run_type == 'test')
            self.min_m = self.i_split
            if run_type == 'val':
                self.max_n = self.j_split
            if run_type == 'test':
                self.min_n = self.j_split

        self.samples = []
        for aoi_id in self.aoi_ids:
            for i in range(self.min_m, self.max_m, self.tiling):
                for j in range(self.min_n, self.max_n, self.tiling):
                    self.samples.append((aoi_id, (i, j)))

        self.m, self.n = self.max_m - self.min_m, self.max_n - self.min_n

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.samples)

    def __getitem__(self, index):

        aoi_id, (i, j) = self.samples[index]

        timestamps = [ts for ts in self.metadata[aoi_id] if not ts['mask']]
        t_values = list(np.linspace(0, len(timestamps), self.T, endpoint=False, dtype=int))
        timestamps = sorted([timestamps[t] for t in t_values], key=lambda ts: int(ts['year']) * 12 + int(ts['month']))

        images = [self.load_planet_mosaic(ts['aoi_id'], ts['dataset'], ts['year'], ts['month']) for ts in timestamps]
        images = np.stack(images)
        if self.add_padding:
            # images = np.pad(images, ((0, 0), (self.tiling, self.tiling), (self.tiling, self.tiling), (0, 0)),
            #                 mode='constant', constant_values=0)
            images = np.pad(images, ((0, 0), (self.tiling, self.tiling), (self.tiling, self.tiling), (0, 0)),
                            mode='reflect')
            i_min, j_min = i, j
            i_max, j_max = i + 3 * self.tiling, j + 3 * self.tiling
            images = images[:, i_min:i_max, j_min:j_max]
        else:
            images = images[:, i:i + self.tiling, j:j + self.tiling]

        labels = [self.load_building_label(aoi_id, ts['year'], ts['month']) for ts in timestamps]
        labels = np.stack(labels)[:, i:i + self.tiling, j:j + self.tiling]

        images, labels = self.transform((images, labels))

        item = {
            'x': images,
            'y': labels,
            'aoi_id': aoi_id,
            'i': i - self.min_m,
            'j': j - self.min_n,
            'dates': [(int(ts['year']), int(ts['month'])) for ts in timestamps],
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Eval {self.name} dataset with {self.length} samples.'


def compose_transformations(cfg, no_augmentations: bool):
    if no_augmentations:
        return transforms.Compose([Numpy2Torch()])

    transformations = []

    # cropping
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'none':
        transformations.append(UniformCrop(cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'change':
        transformations.append(ImportanceRandomCrop(cfg.AUGMENTATION.CROP_SIZE, 'change'))
    elif cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'semantic':
        transformations.append(ImportanceRandomCrop(cfg.AUGMENTATION.CROP_SIZE, 'semantic'))
    else:
        raise Exception('Unkown oversampling type!')

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR_BLUR:
        transformations.append(RandomColorBlur())

    transformations.append(Numpy2Torch())

    if cfg.AUGMENTATION.COLOR_JITTER:
        transformations.append(RandomColorJitter(n_bands=cfg.MODEL.IN_CHANNELS))

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        images, labels = args
        images_tensor = torch.Tensor(images).permute(0, 3, 1, 2)
        labels_tensor = torch.Tensor(labels).permute(0, 3, 1, 2)
        return images_tensor, labels_tensor


class RandomFlip(object):
    def __call__(self, args):
        images, labels = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            images = np.flip(images, axis=2)
            labels = np.flip(labels, axis=2)

        if vertical_flip:
            images = np.flip(images, axis=1)
            labels = np.flip(labels, axis=1)

        images = images.copy()
        labels = labels.copy()

        return images, labels


class RandomRotate(object):
    def __call__(self, args):
        images, labels = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        images = np.rot90(images, k, axes=(1, 2)).copy()
        labels = np.rot90(labels, k, axes=(1, 2)).copy()
        return images, labels


class RandomColorBlur(object):
    def __call__(self, args):
        images, labels = args
        for t in range(images.shape[0]):
            blurred_image = gaussian_filter(images[t], sigma=np.random.rand() / 2)
            images[t] = blurred_image
        return images, labels


class RandomColorJitter(object):
    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, saturation: float = 0.3, hue: float = 0.3,
                 n_bands: int = 3):
        self.cj = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.n_bands = n_bands

    def __call__(self, args):
        images, labels = args
        for t in range(images.shape[0]):
            if self.n_bands <= 3:
                images[t] = self.cj(images[t])
            else:
                # Separate the bands
                bands = [images[t, i] for i in range(self.n_bands)]

                # Apply ColorJitter to each band
                jittered_bands = [self.cj(band.unsqueeze(0)) for band in bands]

                # Stack the bands back together
                jittered_image = torch.cat(jittered_bands, dim=0)
                images[t] = jittered_image
        return images, labels


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def random_crop(self, args):
        images, labels = args
        _, height, width, _ = labels.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        images_crop = images[:, y:y + self.crop_size, x:x + self.crop_size]
        labels_crop = labels[:, y:y + self.crop_size, x:x + self.crop_size]
        return images_crop, labels_crop

    def __call__(self, args):
        images, labels = self.random_crop(args)
        return images, labels


class ImportanceRandomCrop(UniformCrop):
    def __init__(self, crop_size: int, oversampling_type: str):
        super().__init__(crop_size)
        self.oversampling_type = oversampling_type

    def __call__(self, args):

        sample_size = 20
        balancing_factor = 5

        random_crops = [self.random_crop(args) for _ in range(sample_size)]

        if self.oversampling_type == 'change':
            crop_weights = np.array(
                [np.not_equal(crop_label[-1], crop_label[0]).sum() for _, crop_label in random_crops]
            ) + balancing_factor
        elif self.oversampling_type == 'semantic':
            crop_weights = np.array([crop_label.sum() for _, crop_label in random_crops]) + balancing_factor
        else:
            raise Exception('Unkown oversampling type!')

        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(sample_size, p=crop_weights)
        img, label = random_crops[sample_idx]

        return img, label
#
# ---------------------------------------------------------------------------------
# --- ADD ALL THE CODE BELOW TO THE END OF YOUR 'utils/datasets.py' FILE ----------
# ---------------------------------------------------------------------------------
#

#
# ---------------------------------------------------------------------------------
# --- ADD ALL THE CODE BELOW TO THE END OF YOUR 'utils/datasets.py' FILE ----------
# --- (REPLACE any old TSCDDataset classes you added before) ----------------------
# ---------------------------------------------------------------------------------
#

class AbstractTSCDDataset(torch.utils.data.Dataset):
    """
    Abstract Base Class for the PRE-PROCESSED TSCD Dataset.
    This class reads the .jpg structure.
    """

    def __init__(self, cfg: experiment_manager.CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.timestamps = [2016, 2018, 2020, 2022]
        self.T = cfg.DATALOADER.TIMESERIES_LENGTH
        assert self.T == 4, "TSCD config must have TIMESERIES_LENGTH: 4"
        self.name = cfg.DATASET.NAME
        
        # Get the edge definitions from the helper
        self.edges_dense = helpers.get_edges(self.T, 'dense')
        self.edges_adj = helpers.get_edges(self.T, 'adjacent')

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def load_tscd_img(self, dataset: str, index: str, year: int) -> np.ndarray:
        """ Loads a single image tile from the pre-processed folder """
        file = self.root_path / dataset / 'imgs' / f'{index}_{year}.jpg'
        # Load with OpenCV
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        # Convert BGR (OpenCV default) to RGB (PyTorch default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 # Normalize
        return img.astype(np.float32)

    def load_adjacent_change_label(self, dataset: str, index: str, year_t1: int, year_t2: int) -> np.ndarray:
        """ Loads a single ADJACENT change label tile from the pre-processed folder """
        file = self.root_path / dataset / 'change_labels_adj' / f'{index}_{year_t1}-{year_t2}.jpg'
        # Load as grayscale
        label = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if label.ndim == 2:
            label = label[:, :, None] # Add channel dim
        label = label > 0 # Binarize
        return label.astype(np.float32)

    def derive_dense_labels(self, adj_labels: list) -> np.ndarray:
        """
        Derives dense change labels from adjacent ones, as per paper [53].
        "An odd number of adjacent changes indicates a change..."
        This is a logical XOR (^) operation.
        """
        ch_16_18, ch_18_20, ch_20_22 = adj_labels
        
        # Calculate non-adjacent labels
        ch_16_20 = np.logical_xor(ch_16_18, ch_18_20)
        ch_18_22 = np.logical_xor(ch_18_20, ch_20_22)
        ch_16_22 = np.logical_xor(ch_16_20, ch_20_22)
        
        # self.edges_dense is: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # This MUST match the order expected by the model
        dense_labels = np.stack([
            ch_16_18,    # Edge (0, 1) -> 2016-2018
            ch_16_20,    # Edge (0, 2) -> 2016-2020
            ch_16_22,    # Edge (0, 3) -> 2016-2022
            ch_18_20,    # Edge (1, 2) -> 2018-2020
            ch_18_22,    # Edge (1, 3) -> 2018-2022
            ch_20_22     # Edge (2, 3) -> 2020-2022
        ], axis=0) # Shape: (6, H, W, 1)
        
        return dense_labels.astype(np.float32)


class TrainTSCDDataset(AbstractTSCDDataset):
    """
    Training Dataset Class for TSCD.
    """
    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.no_augmentations = no_augmentations
        self.transform = compose_transformations(cfg, no_augmentations)

        self.dataset = 'train' # Hardcoded for training
        if run_type == 'val':
             self.dataset = 'val'
        
        json_file = self.root_path / f'samples_{self.dataset}.json'
        self.samples = helpers.load_json(json_file)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.length = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        index_str = sample['index'] # e.g., "0", "1"

        # 1. Load the 4 images
        images = [self.load_tscd_img(self.dataset, index_str, year) for year in self.timestamps]
        images = np.stack(images) # (4, H, W, 3)

        # 2. Load the 3 adjacent change labels
        adj_labels = []
        for (t1, t2) in self.edges_adj:
            year1 = self.timestamps[t1]
            year2 = self.timestamps[t2]
            adj_labels.append(self.load_adjacent_change_label(self.dataset, index_str, year1, year2))
        
        # 3. Derive the 6 dense change labels
        labels_ch = self.derive_dense_labels(adj_labels) # (6, H, W, 1)

        # 4. Apply augmentations
        images, labels_ch = self.transform((images, labels_ch))

        item = {
            'x': images,      # (4, C, H, W)
            'y_ch': labels_ch # (6, 1, H, W)
        }
        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Train {self.name} dataset ({self.dataset}) with {self.length} samples.'


#
# --- REPLACE the old EvalTSCDDataset in utils/datasets.py with this one ---
#
class EvalTSCDDataset(AbstractTSCDDataset):
    """
    Evaluation Dataset Class for TSCD. (For Validation)
    This version is adaptive and does not assume 512x512.
    """
    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, tiling: int = None, add_padding: bool = False):
        super().__init__(cfg)
        
        self.tiling = tiling
        self.add_padding = add_padding
        self.transform = compose_transformations(cfg, no_augmentations=True)

        assert run_type == 'val', "Use EvalTSCDDataset for 'val' split only."
        self.dataset = 'val'
        
        json_file = self.root_path / f'samples_{self.dataset}.json'
        samples = helpers.load_json(json_file)

        if tiling is None:
            self.tiling = 512 # Default tile size
        
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE # This is 64

        self.samples = []
        
        # --- START OF FIX ---
        # We must load one image to get the real dimensions
        # We can't assume 512x512
        if len(samples) > 0:
            first_sample_idx = samples[0]['index']
            # Load the first image to check its shape
            temp_img = self.load_tscd_img(self.dataset, first_sample_idx, self.timestamps[0])
            height, width, _ = temp_img.shape
            print(f"EvalTSCDDataset: Detected image size as {height}x{width}. Creating tiles...")
        else:
            height, width = 0, 0
            
        for sample in samples:
            # Create tiles based on ACTUAL image size
            for i in range(0, height, self.crop_size): 
                for j in range(0, width, self.crop_size):
                    # Check if crop is out of bounds
                    if (i + self.crop_size > height) or (j + self.crop_size > width):
                        continue
                    tile_sample = {
                        'index': sample['index'],
                        'i': i,
                        'j': j,
                    }
                    self.samples.append(tile_sample)
        # --- END OF FIX ---

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.length = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        index_str, i, j = sample['index'], sample['i'], sample['j']

        # 1. Load 4 images
        images = [self.load_tscd_img(self.dataset, index_str, year) for year in self.timestamps]
        images = np.stack(images) # (4, H, W, 3)
        
        # 2. Load 3 adjacent labels
        adj_labels = []
        for (t1, t2) in self.edges_adj:
            year1 = self.timestamps[t1]
            year2 = self.timestamps[t2]
            adj_labels.append(self.load_adjacent_change_label(self.dataset, index_str, year1, year2))
        
        # 3. Derive 6 dense labels
        labels_ch = self.derive_dense_labels(adj_labels) # (6, H, W, 1)

        # 4. Crop the full tiles
        images = images[:, i:i + self.crop_size, j:j + self.crop_size]
        labels_ch = labels_ch[:, i:i + self.crop_size, j:j + self.crop_size]

        # 5. Apply transforms (Numpy2Torch)
        images, labels_ch = self.transform((images, labels_ch))

        item = {
            'x': images,
            'y_ch': labels_ch, 
            'y': torch.empty(0), # Add empty tensor for 'y'
            'index': index_str,
            'i': i,
            'j': j,
        }
        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Eval {self.name} dataset with {self.length} samples.'
#
# --- ADD THIS CLASS TO THE VERY END OF utils/datasets.py ---
#
class EvalTestTSCDDataset(AbstractTSCDDataset):
    """
    Test Evaluation Dataset Class for TSCD.
    Based on EvalTestWUSUDataset and EvalTSCDDataset.
    """
    def __init__(self, cfg: experiment_manager.CfgNode, site: str = None):
        super().__init__(cfg)

        # You must create this metadata_test.json file for your test set
        # For now, we'll just read the 'samples_test.json'
        # metadata_file = self.root_path / 'metadata_test.json'
        # self.metadata = helpers.load_json(metadata_file)
        
        self.dataset = 'test'
        json_file = self.root_path / f'samples_{self.dataset}.json'
        samples = helpers.load_json(json_file)

        self.transform = compose_transformations(cfg, no_augmentations=True)
        self.crop_size = cfg.AUGMENTATION.CROP_SIZE # This is 64

        self.samples = []
        
        # --- START OF FIX ---
        # Load one image to get the real dimensions
        if len(samples) > 0:
            first_sample_idx = samples[0]['index']
            # Load the first image to check its shape
            temp_img = self.load_tscd_img(self.dataset, first_sample_idx, self.timestamps[0])
            height, width, _ = temp_img.shape
            print(f"EvalTestTSCDDataset: Detected image size as {height}x{width}. Creating tiles...")
        else:
            height, width = 0, 0
            
        for sample in samples:
            # Create tiles based on ACTUAL image size
            for i in range(0, height, self.crop_size): 
                for j in range(0, width, self.crop_size):
                    # Check if crop is out of bounds
                    if (i + self.crop_size > height) or (j + self.crop_size > width):
                        continue
                    tile_sample = {
                        'index': sample['index'],
                        'i': i,
                        'j': j,
                    }
                    self.samples.append(tile_sample)
        # --- END OF FIX ---

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.length = len(self.samples)

    def __getitem__(self, sample_index):
        sample = self.samples[sample_index]
        index_str, i, j = sample['index'], sample['i'], sample['j']

        # 1. Load 4 images
        images = [self.load_tscd_img(self.dataset, index_str, year) for year in self.timestamps]
        images = np.stack(images) # (4, H, W, 3)
        
        # 2. Load 3 adjacent labels
        adj_labels = []
        for (t1, t2) in self.edges_adj:
            year1 = self.timestamps[t1]
            year2 = self.timestamps[t2]
            adj_labels.append(self.load_adjacent_change_label(self.dataset, index_str, year1, year2))
        
        # 3. Derive 6 dense labels
        labels_ch = self.derive_dense_labels(adj_labels) # (6, H, W, 1)

        # 4. Crop the full tiles
        images = images[:, i:i + self.crop_size, j:j + self.crop_size]
        labels_ch = labels_ch[:, i:i + self.crop_size, j:j + self.crop_size]

        # 5. Apply transforms (Numpy2Torch)
        images, labels_ch = self.transform((images, labels_ch))

        item = {
            'x': images,
            'y_ch': labels_ch, 
            'y': torch.empty(0), # Add empty tensor for 'y'
            'index': index_str,
            'i': i,
            'j': j,
        }
        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Eval Test {self.name} dataset with {self.length} samples.'