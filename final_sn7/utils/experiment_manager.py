import argparse
from argparse import ArgumentParser
import yaml
from fvcore.common.config import CfgNode as _CfgNode
from pathlib import Path

class CfgNode(_CfgNode):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        self.__dict__[CfgNode.NEW_ALLOWED] = True
        super(CfgNode, self).__init__(init_dict, key_list, True)

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

def new_config():
    C = CfgNode()
    C.CONFIG_DIR = 'configs/'
    C.PATHS = CfgNode()
    C.TRAINER = CfgNode()
    
    # Define exact tree to prevent AttributeErrors
    C.MODEL = CfgNode()
    C.MODEL.TRANSFORMER_PARAMS = CfgNode()
    C.MODEL.REFINER = CfgNode()
    C.MODEL.LOSS = CfgNode()
    
    C.DATALOADER = CfgNode()
    C.AUGMENTATION = CfgNode()
    C.DATASET = CfgNode()
    C.CONSISTENCY_TRAINER = CfgNode()

    return C.clone()

def setup_cfg(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    if hasattr(args, 'opts') and args.opts:
        cfg.merge_from_list(args.opts)
        
    cfg.NAME = args.config_file
    cfg.PATHS.ROOT = str(Path.cwd())
    
    # Fix: Auto-create directory instead of asserting
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.PATHS.OUTPUT = str(out_dir)
    
    assert Path(args.dataset_dir).exists(), f"Dataset dir {args.dataset_dir} not found!"
    cfg.PATHS.DATASET = args.dataset_dir
    return cfg

def load_cfg(config_name: str):
    cfg = new_config()
    cfg_file = Path.cwd() / 'configs' / f'{config_name}.yaml'
    cfg.merge_from_file(str(cfg_file))
    cfg.NAME = config_name
    return cfg