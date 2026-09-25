import json
from pathlib import Path
from easydict import EasyDict as edict


def get_config_regression(model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config.json"
    with open(config_file, 'r', encoding='utf-8') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    dataset_root = Path(__file__).parent / config_all['datasetCommonParams']['dataset_root_dir']
    config['featurePath'] = str((dataset_root / config['featurePath']).resolve())
    local_pretrained = Path(__file__).parent / config['pretrained']
    if local_pretrained.is_dir():
        config['pretrained'] = str(local_pretrained.resolve())
    config = edict(config)

    return config
