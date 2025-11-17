import torch

from utils import datasets, parsers, experiment_manager, helpers, evaluation
from utils.experiment_manager import CfgNode
from model import model

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assessment(cfg: CfgNode, edge_type: str = 'dense', run_type: str = 'test'):
    print(f"Running evaluation for: {cfg.NAME}")
    print(f"Loading model from: {cfg.PATHS.OUTPUT}")
    
    net = model.load_model(cfg, device)
    print("Model loaded successfully.")
    
    # Run the evaluation on the specified split (e.g., 'test')
    m = evaluation.run_quantitative_evaluation(net, cfg, device, run_type, enable_mti=True, mti_edge_setting=edge_type)

    data = {}
    
    # --- START MODIFICATION ---
    # Add conditional logic for TSCD dataset
    if cfg.DATASET.NAME == 'tscd':
        print(f"\n--- TSCD Test Set Results (F1 / IoU) ---")
        for attr in ['ch_cont', 'ch_fl']:
            f1 = evaluation.f1_score(getattr(m, f'TP_{attr}'), getattr(m, f'FP_{attr}'), getattr(m, f'FN_{attr}'))
            iou = evaluation.iou(getattr(m, f'TP_{attr}'), getattr(m, f'FP_{attr}'), getattr(m, f'FN_{attr}'))
            data[attr] = {'f1': f1, 'iou': iou}
            # Print the final scores
            print(f"F1 Score ({attr}): {f1:.4f}")
            print(f"IoU ({attr}):      {iou:.4f}")
            
    else:
        # Original logic for other datasets
        print(f"\n--- {cfg.DATASET.NAME} Test Set Results (F1 / IoU) ---")
        for attr in ['seg_cont', 'seg_fl', 'ch_cont', 'ch_fl']:
            f1 = evaluation.f1_score(getattr(m, f'TP_{attr}'), getattr(m, f'FP_{attr}'), getattr(m, f'FN_{attr}'))
            iou = evaluation.iou(getattr(m, f'TP_{attr}'), getattr(m, f'FP_{attr}'), getattr(m, f'FN_{attr}'))
            data[attr] = {'f1': f1, 'iou': iou}
            print(f"F1 Score ({attr}): {f1:.4f}")
    # --- END MODIFICATION ---

    eval_folder = Path(cfg.PATHS.OUTPUT) / 'evaluation'
    eval_folder.mkdir(exist_ok=True)
    json_path = eval_folder / f'{cfg.NAME}_{edge_type}_{run_type}.json'
    helpers.write_json(json_path, data)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # Force the run_type to 'test' to evaluate the test set
    assessment(cfg, edge_type=args.edge_type, run_type='test')
    