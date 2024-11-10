"""
Add support for T3A Wrapper
"""
import os
import torch

import argparse
from pathlib import Path
import numpy as np
import glob
import shutil

from datasets import DataInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpus')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)
    # pl.seed_everything(41)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'external_test': cfg.Data.external,
                            }
    if cfg.Data.get('survival', False):
        from models.model_interface_survival import ModelInterface
        model = ModelInterface(**ModelInterface_dict)
    else:
        from models import ModelInterface
        model = ModelInterface(**ModelInterface_dict)
    
    shutil.copy(os.path.join('models', f'{cfg.Model.name}.py'), cfg.log_path)

    deterministic_flag = False
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus, 
        deterministic=deterministic_flag,
        check_val_every_n_epoch=1,
        gradient_clip_val= cfg.Optimizer.grad_clip if cfg.Optimizer.grad_clip else 0,
    )
    
    if not deterministic_flag:
        pass
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True, warn_only=True)
        

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if cfg.General.monitor in str(model_path)]

        best_metric = 0 if cfg.General.mode == 'max' else 0x3f3f3f3f # big number

        best_model_path = None
        for model_path in model_paths:
            metric = float(model_path.split(f'{cfg.General.monitor}=')[1].split('.ckpt')[0].split('v')[0].split('-')[0])
            print(f'{metric=},{best_metric=}')
            if cfg.General.mode == 'max':
                if metric > best_metric:
                    best_metric = metric
                    best_model_path = model_path
            elif cfg.General.mode == 'min':
                if metric < best_metric:
                    best_metric = metric
                    best_model_path = model_path
            else:
                raise NotImplementedError


        trainer = Trainer(
            num_sanity_val_steps=0, 
            logger=None,
            callbacks=cfg.callbacks,
            max_epochs= cfg.General.epochs,
            gpus=cfg.General.gpus, 
            deterministic=deterministic_flag,
            check_val_every_n_epoch=1,
            gradient_clip_val= cfg.Optimizer.grad_clip if cfg.Optimizer.grad_clip else 0,
        )

        for path in [best_model_path]:
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg, log=cfg.log_path) # LightningModule
            
            trainer.test(model=new_model, datamodule=dm)
            # one trainer instance cannot be used twice
            break
        
        if len(model_paths) > 1:
            print(f'\n\033[1;31mMultiple Checkpoints found, only using the best {best_model_path}!\033[0m\n')

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)
            
    #---->update
    cfg.Data.fold = args.fold
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    if args.seed:
        cfg.General.seed = args.seed

    #---->main
    main(cfg)
 