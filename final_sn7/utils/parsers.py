import argparse

def training_argument_parser():
    parser = argparse.ArgumentParser(description="Training Argument Parser")
    
    # Maps -c to config_file
    parser.add_argument("-c", "--config", dest="config_file", type=str, required=True, help="Path to config file")
    
    # Maps -o to output_dir
    parser.add_argument("-o", "--output", dest="output_dir", type=str, required=True, help="Output directory")
    
    # Maps -d to dataset_dir
    parser.add_argument("-d", "--dataset", dest="dataset_dir", type=str, required=True, help="Dataset directory name")
    
    # Handles the weighted checkpoint
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)
    return parser

def inference_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
    parser.add_argument('-e', "--edge-type", dest='edge_type', default='dense', help="mrf edge type")
    parser.add_argument('-o', "--output-dir", dest='output_dir', required=True, help="path to output directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to output directory")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')