import argparse
import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import DLF
import sys

from datetime import datetime      
now = datetime.now()
format = "%Y/%m/%d %H:%M:%S"
formatted_now = now.strftime(format)
formatted_now = str(formatted_now)+" - "

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DLF_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=1, verbose_level=1, mode = '', is_training = False 
):
    # Initialization
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()
    
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    

    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_training = is_training  
    args.mode = mode # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)


    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    if args.is_training:
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}-valid.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time"]+["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        
        res = [formatted_now]+res 
        df.loc[len(df)] = res    
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False): 

    dataloader = MMDataLoader(args, num_workers, include_test=(args.mode == 'test'))

    if args.is_training:
        print("training for DLF")

        
        model = []
        model_DLF = getattr(DLF, 'DLF')(args)
        if getattr(args, 'init_checkpoint', None):
            source = Path(args.init_checkpoint)
            if not source.is_file():
                raise FileNotFoundError(f'Initialization checkpoint not found: {source}')
            missing, unexpected = model_DLF.load_state_dict(
                torch.load(source, map_location='cpu'), strict=False)
            logger.info('Initialized shared backbone from %s (%d new, %d unused keys)',
                        source, len(missing), len(unexpected))

        model_DLF = model_DLF.cuda()

        model = [model_DLF]         
    else:
        print("testing phase for DLF")
        model = getattr(DLF, 'DLF')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)


    #test
    if args.mode == 'test':
        model.load_state_dict(torch.load(args.model_save_path, map_location=args.device), strict=True)
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
    #train
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load(args.model_save_path, map_location=args.device))

        results = trainer.do_test(model[0], dataloader['valid'], mode="VAL")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results


def main():
    """Train the final coalition-aware model on attachment 2."""
    parser = argparse.ArgumentParser(description=__doc__ or 'Train final DLF model')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--check-config', action='store_true',
                        help='Check training inputs and exit without changing weights.')
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    init_checkpoint = root / 'checkpoints/transfer_init/model.pth'
    final_checkpoint = root / 'checkpoints/final/model.pth'
    training_config = {
        'adaptive_main': True,
        'coalition_aware': True,
        'coalition_loss_weight': 0.2,
        'gate_temperature': 1.0,
        'init_checkpoint': str(init_checkpoint),
        'model_save_path': str(final_checkpoint),
        'train_fusion_only': True,
        'learning_rate': 3e-4,
        'max_epochs': 5,
        'early_stop': 2,
    }
    if cli.check_config:
        args = get_config_regression('DLF', 'mosei')
        required = [init_checkpoint, Path(args.featurePath), Path(args.pretrained)]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing training inputs: ' + ', '.join(missing))
        print('Training inputs OK; seed=', cli.seed,
              '; attachment-2 features=', args.featurePath,
              '; initialization=', init_checkpoint,
              '; output=', final_checkpoint)
        return
    DLF_run(
        model_name='DLF', dataset_name='mosei', config=training_config,
        is_tune=False, seeds=[cli.seed],
        model_save_dir=str(root / 'checkpoints/final'),
        res_save_dir=str(root / 'outputs/training'),
        log_dir=str(root / 'outputs/logs'),
        mode='train', is_training=True)


if __name__ == '__main__':
    main()
