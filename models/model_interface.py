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
from sklearn.metrics import balanced_accuracy_score

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
        print(f'self.log_path: {self.log_path}')
        
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            self.train_AUC = torchmetrics.AUROC(num_classes=self.n_classes, average = 'macro')
            self.test_AUC = torchmetrics.AUROC(num_classes=self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes = self.n_classes, average='micro'),
                torchmetrics.F1Score(num_classes = self.n_classes, average = 'macro'),
                torchmetrics.Recall(average = 'macro', num_classes = self.n_classes),
                torchmetrics.Precision(average = 'macro', num_classes = self.n_classes),
                torchmetrics.Specificity(average = 'macro', num_classes = self.n_classes)
            ])
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro')
            self.train_AUC = torchmetrics.AUROC(num_classes=2, average = 'macro')
            self.test_AUC = torchmetrics.AUROC(num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes = 2, average = 'micro'),
                torchmetrics.F1Score(num_classes = 2, average = 'macro'),
                torchmetrics.Recall(average = 'macro', num_classes = 2),
                torchmetrics.Precision(average = 'macro', num_classes = 2)])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0

        self.label_dict = kargs['data'].label_dict


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch['data'], batch['label']

        if 'CLAM' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, instance_eval=True)
        else:
            results_dict = self.model(data=data, label=label)
        
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = results_dict['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=1)
        
        self.train_AUC.update(Y_prob, label.squeeze())
        self.train_metrics.update(Y_prob, label)

        #---->acc log
            
        for i in range(len(Y_hat)):
            Y_hat_i = int(Y_hat[i])
            Y_i = int(label[i])
            self.data[Y_i]["count"] += 1
            self.data[Y_i]["correct"] += (Y_hat_i == Y_i)

        return {'loss': loss} 

    # Validation is in the last of the training epoch
    # i.e., run_validation_epoch is before training_epoch_end
    def on_validation_start(self):
        try:
            print("\n\n\n====\033[1;32mTraining\033[0m Statistics====")
            print(f'\033[1;34mLog Path\033[0m: {self.log_path}')
            for c in range(self.n_classes):
                count = self.data[c]["count"]
                correct = self.data[c]["correct"]
                if count == 0: 
                    acc = 0
                else:
                    acc = float(correct) / count
                print('class \033[1;34m{}\033[0m: acc \033[1;31m{:.4f}\033[0m, correct \033[1;31m{}\033[0m/\033[1;35m{}\033[0m'.format(c, acc, correct, count))
            train_auc = self.train_AUC.compute()
            self.train_AUC.reset()
            print('\033[1;34mTrain AUC\033[0m: \033[1;31m{:.4f}\033[0m'.format(train_auc))
            self.log('train_auc', train_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)

            train_metrics_dict = self.train_metrics.compute()
            self.train_metrics.reset()
            self.log_dict(train_metrics_dict, on_step=False, on_epoch=True, logger=True, batch_size=1)
            print('\033[1;34mTrain Balanced Accuracy\033[0m: \033[1;31m{:.4f}\033[0m'.format(train_metrics_dict['train_Recall']))
            print('\n')
            
            self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        except:
            pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, label, slide_id = batch['data'], batch['label'], batch['slide_id']
        if 'CLAM' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, instance_eval=False)
        elif 'Mamba' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, inference=True)
        else:
            results_dict = self.model(data=data, label=label)
        
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        loss = results_dict['loss'].item()

        if dataloader_idx == 0 or dataloader_idx == None:
            #---->acc log
            for i in range(len(Y_hat)):
                Y_hat_i = int(Y_hat[i])
                Y_i = int(label[i])
                self.data[Y_i]["count"] += 1
                self.data[Y_i]["correct"] += (Y_hat_i == Y_i)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'loss': loss, 'slide_id': slide_id}
        


    def validation_epoch_end(self, val_step_outputs):
        if isinstance(val_step_outputs[0], dict): # only one dataloader
            val_step_outputs = [val_step_outputs]

        for idx, output in enumerate(val_step_outputs):
            mode = ['val', 'test'][idx]

            logits = torch.cat([x[f'logits'] for x in output], dim = 0)
            probs = torch.cat([x[f'Y_prob'] for x in output], dim = 0)
            max_probs = torch.cat([x[f'Y_hat'] for x in output], dim=0)
            target = torch.cat([x[f'label'] for x in output], dim = 0)
            losses = [x[f'loss'] for x in output]
            
            #---->
            if mode == 'val': # validation
                val_auc = self.AUROC(probs, target.squeeze())
                self.AUROC.reset()
                metrics_dict = self.valid_metrics(max_probs.squeeze() , target.squeeze())
                self.valid_metrics.reset()
                self.log(f'val_auc', val_auc, on_epoch=True, logger=True, batch_size=1)
            else: # test
                # if isinstance(output[0]['slide_id'], torch.Tensor):
                #     slide_id = np.concatenate([x['slide_id'].cpu().numpy() for x in output], axis=0)
                # else:
                #     slide_id = np.concatenate([x['slide_id'] for x in output], axis=0)
                # pprint(f'{list(zip(slide_id,probs.cpu().numpy()[:,1]))}')
                test_auc = self.test_AUC(probs, target.squeeze())
                self.test_AUC.reset()
                metrics_dict = self.test_metrics(max_probs.squeeze() , target.squeeze())
                self.test_metrics.reset()
                self.log(f'test_auc', test_auc, on_epoch=True, logger=True, batch_size=1)
                
            self.log(f'{mode}_loss', np.mean(losses), prog_bar=True, on_epoch=True, logger=True, batch_size=1)
            self.log_dict(metrics_dict, on_epoch = True, logger = True, batch_size=1)

            if mode == 'val':
                print("\n\n\n====\033[1;31mValidation\033[0m Statistics====")
                print(f'\033[1;34mLog Path\033[0m: {self.log_path}')
                #---->acc log
                for c in range(self.n_classes):
                    count = self.data[c]["count"]
                    correct = self.data[c]["correct"]
                    if count == 0: 
                        acc = 0
                    else:
                        acc = float(correct) / count
                    print('class \033[1;34m{}\033[0m: acc \033[1;31m{:.4f}\033[0m, correct \033[1;31m{}\033[0m/\033[1;35m{}\033[0m'.format(c, acc, correct, count))
                print('\033[1;34mValidation AUC\033[0m: \033[1;31m{:.4f}\033[0m'.format(val_auc))
                print('\033[1;34mValidation Avearge Recall\033[0m: \033[1;31m{:.4f}\033[0m'.format(metrics_dict['val_Recall']))
                print('\n')
            else:
                print("====\033[1;31mTest\033[0m Statistics====")
                print(f'\033[1;34mLog Path\033[0m: {self.log_path}')
                print('\033[1;34mTest AUC\033[0m: \033[1;31m{:.4f}\033[0m'.format(test_auc))
                print('\033[1;34mTest Avearge Recall\033[0m: \033[1;31m{:.4f}\033[0m'.format(metrics_dict['test_Recall']))
                print('\n')

        
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)
    

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return optimizer

    def test_step(self, batch, batch_idx):
        data, label, slide_id = batch['data'], batch['label'], batch['slide_id']
        
        if 'CLAM' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, instance_eval=False)
        elif 'Mamba' in self.hparams.model.name:
            results_dict = self.model(data=data, label=label, inference=True)
        else:
            results_dict = self.model(data=data, label=label)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        for i in range(len(Y_hat)):
            Y_hat_i = int(Y_hat[i])
            Y_i = int(label[i])
            self.data[Y_i]["count"] += 1
            self.data[Y_i]["correct"] += (Y_hat_i == Y_i)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'slide_id': slide_id}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.cat([x['Y_hat'] for x in output_results], dim = 0)
        target = torch.cat([x['label'] for x in output_results], dim = 0)
        if isinstance(output_results[0]['slide_id'], torch.Tensor):
            slide_id = np.concatenate([x['slide_id'].cpu().numpy() for x in output_results], axis=0)
        else:
            slide_id = np.concatenate([x['slide_id'] for x in output_results], axis=0)
        # pprint(f'{list(zip(slide_id,probs.cpu().numpy()[:,0]))}')
        
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['test_auc'] = auc
        for keys, values in metrics.items():
            print(f'\033[1;34m{keys}\033[0m = \033[1;31m{values:.4f}\033[0m')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class \033[1;34m{}\033[0m: acc \033[1;31m{:.4f}\033[0m, correct \033[1;31m{}\033[0m/\033[1;35m{}\033[0m'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        if len(max_probs.shape) == 1:
            case_result = pd.DataFrame({'slide_id': slide_id, 'pred_label': max_probs.cpu().numpy(), 'pred_probs': probs[:,1].cpu().numpy(), 'GT': target.cpu().numpy()})
        else:
            case_result = pd.DataFrame({'slide_id': slide_id, 'pred_label': max_probs[:,0].cpu().numpy(), 'pred_probs': probs[:,1].cpu().numpy(), 'GT': target.cpu().numpy()})

        result.to_csv(self.log_path / 'result.csv')
        case_result.to_csv(self.log_path / f'case.csv')
        
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
