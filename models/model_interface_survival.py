import sys
import copy
import numpy as np
import inspect
import importlib
import random
import pandas as pd


#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from sksurv.metrics import concordance_index_censored

#---->
import torch
import torchmetrics

#---->
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    import lightning as pl
from pprint import pprint

    
class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        self.train_risk_scores = []
        self.train_censorship = []
        self.train_event_time = []

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label, c, event_time = batch['data'], batch['label'], batch['c'], batch['event_time']

        if 'CLAM' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, c=c, instance_eval=True)
        else:
            results_dict = self.model(data=data, label=label, c=c, )
        
        S = results_dict['S']

        #---->loss
        loss = results_dict['loss']
        self.log('train_loss', loss, batch_size=1, on_step=True, on_epoch=True, logger=True)
        
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()

        self.train_risk_scores.append(risk.item())
        self.train_censorship.append(c.item())
        self.train_event_time.append(event_time.item())

        return {'loss': loss} 

    # Validation is in the last of the training epoch
    # i.e., run_validation_epoch is before training_epoch_end
    def on_validation_start(self):
        print("\n\n\n====\033[1;32mTraining\033[0m Statistics====")
        print(f'\033[1;34mLog Path\033[0m: {self.log_path}')

        self.train_risk_scores = np.array(self.train_risk_scores)   
        self.train_censorship = np.array(self.train_censorship)
        self.train_event_time = np.array(self.train_event_time)

        c_index = concordance_index_censored((1-self.train_censorship).astype(bool), self.train_event_time, self.train_risk_scores, tied_tol=1e-08)[0]
        self.log('train_c_index', c_index, batch_size=1, on_epoch=True, prog_bar=True, logger=True)

        print('\033[1;34mTraining C-Index\033[0m: \033[1;31m{:.4f}\033[0m'.format(c_index))
        
        self.train_risk_scores = []
        self.train_censorship = []
        self.train_event_time = []


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, label, c, event_time = batch['data'], batch['label'], batch['c'], batch['event_time']
        if 'CLAM' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, c=c, instance_eval=False)
        elif 'Mamba' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, c=c, inference=True)
        else:
            results_dict = self.model(data=data, label=label, c=c)

        loss = results_dict['loss']
        
        S = results_dict['S']
        risk = -torch.sum(S, dim=1)


        return {'risk' : risk, 'censorship' : c, 'event_time' : event_time, 'label' : label, 'loss': loss.item()}


    def validation_epoch_end(self, val_step_outputs):
        if isinstance(val_step_outputs[0], dict): # only one dataloader
            val_step_outputs = [val_step_outputs]

        for idx, output in enumerate(val_step_outputs):
            mode = ['val', 'test'][idx]

            all_risk_scores = torch.cat([x[f'risk'] for x in output], dim = 0)
            all_censorships = torch.cat([x[f'censorship'] for x in output], dim = 0)
            all_event_times = torch.cat([x[f'event_time'] for x in output], dim=0)
            c_index = concordance_index_censored((1-all_censorships).cpu().numpy().astype(bool), all_event_times.cpu().numpy(), all_risk_scores.cpu().numpy(), tied_tol=1e-08)[0]

            self.log(f'{mode}_c_index', c_index, batch_size=1, on_epoch=True, prog_bar=True, logger=True)
            losses = [x[f'loss'] for x in output]
            self.log(f'{mode}_loss', np.mean(losses), batch_size=1, prog_bar=True, on_epoch=True, logger=True)

            if mode == 'val':
                print("\n\n\n====\033[1;31mValidation\033[0m Statistics====")
                print(f'\033[1;34mLog Path\033[0m: {self.log_path}')
                print('\033[1;34mValidation C-Index\033[0m: \033[1;31m{:.4f}\033[0m'.format(c_index))
                print('\n')
            else:
                print("====\033[1;31mTest\033[0m Statistics====")
                print(f'\033[1;34mLog Path\033[0m: {self.log_path}')
                print('\033[1;34mTest C-Index\033[0m: \033[1;31m{:.4f}\033[0m'.format(c_index))
                print('\n')
    
    def test_step(self, batch, batch_idx):
        data, label, c, event_time = batch['data'], batch['label'], batch['c'], batch['event_time']
        
        if 'CLAM' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, c=c, instance_eval=False)
        elif 'Mamba' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, c=c, inference=True)
        else:
            results_dict = self.model(data=data, label=label, c=c)

        loss = results_dict['loss']
        
        S = results_dict['S']
        risk = -torch.sum(S, dim=1)


        return {'risk' : risk, 'censorship' : c, 'event_time' : event_time, 'label' : label, 'loss': loss.item(), 'label' : label}

    def test_epoch_end(self, output_results):
        all_risk_scores = torch.cat([x[f'risk'] for x in output_results], dim = 0)
        all_censorships = torch.cat([x[f'censorship'] for x in output_results], dim = 0)
        all_event_times = torch.cat([x[f'event_time'] for x in output_results], dim=0)
        c_index = concordance_index_censored((1-all_censorships).cpu().numpy().astype(bool), all_event_times.cpu().numpy(), all_risk_scores.cpu().numpy(), tied_tol=1e-08)[0]

        self.log(f'test_c_index', c_index, batch_size=1, on_epoch=True, prog_bar=True, logger=True)
        losses = [x[f'loss'] for x in output_results]
        self.log(f'test_loss', np.mean(losses), batch_size=1, prog_bar=True, on_epoch=True, logger=True)

        print("====\033[1;31mTest\033[0m Statistics====")
        print(f'\033[1;34mLog Path\033[0m: {self.log_path}')
        print('\033[1;34mTest C-Index\033[0m: \033[1;31m{:.4f}\033[0m'.format(c_index))
        print('\n')

        with open(self.log_path / 'result.txt', 'w') as f:
            f.write(f'Test C-Index: {c_index}\n')
        

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return optimizer

    def load_model(self):
        name = self.hparams.model.name
        if name == 'MamMIL2V15':
            name = 'MamMIL2'
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name! ' + camel_name)
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)
