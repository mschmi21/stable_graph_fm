from .fmri import FMRIData
from .sbm import SBMData

def get_dataset(config: dict, split_seed: int = 0):
    name = config['dataset']['name']
    
    if name == "fmri":
        dataset = FMRIData(
            data_path=config['dataset']['data_path'],
            train_prop=config['dataset']['train_prop'],
            val_prop=config['dataset']['val_prop'],
            split_seed=split_seed
        )
        dataset.prepare_data()
        return dataset
    
    if name == "sbm":
        dataset = SBMData(
            sizes=config['dataset']['sizes'],
            p_in=config['dataset']['p_in'],
            p_out=config['dataset']['p_out'],
            num_samples=config['dataset']['num_samples'],
            train_prop=config['dataset']['train_prop'],
            val_prop=config['dataset']['val_prop'], 
            split_seed=split_seed
        )
        dataset.prepare_data()
        return dataset
    
    raise ValueError(f"Dataset '{name}' is not supported.")