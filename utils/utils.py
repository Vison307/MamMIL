from pathlib import Path

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
try:
    from pytorch_lightning import loggers as pl_loggers
except ModuleNotFoundError:
    from lightning.pytorch import loggers as pl_loggers

def load_loggers(cfg):

    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = Path(cfg.config).parent 
    version_name = Path(cfg.config).name[:-5] + f'/s{cfg.General.seed}'
    cfg.log_path = Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'
    print(f'---->Log dir: {cfg.log_path}')
    
    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(str(Path(log_path) / log_name),
                                             name = version_name, version = f'fold{cfg.Data.fold}',
                                             log_graph = False, default_hp_metric = False)
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(str(Path(log_path) / log_name),
                                      name = version_name, version = f'fold{cfg.Data.fold}')
    
    return [tb_logger, csv_logger]


#---->load 
try:
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ModuleNotFoundError:
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
        
def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    if cfg.General.patience > 0:
        early_stop_callback = EarlyStopping(
            monitor=cfg.General.monitor,
            min_delta=0.00,
            patience=cfg.General.patience,
            verbose=True,
            mode=cfg.General.mode,
        )
        
        Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = cfg.General.monitor,
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}-{'+cfg.General.monitor+':.4f}-{test_auc:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = cfg.General.mode,
                                         save_weights_only = True))
    Mycallbacks.append(LearningRateMonitor())
    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=-1) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss
