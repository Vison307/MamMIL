import inspect # 查看python 类的参数和模块、函数代码
import importlib # In order to dynamically import the library
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    import lightning as pl
import torch
import os

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

class DataInterface(pl.LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1, dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()
        
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        
        self.kwargs = kwargs

        self.survival = kwargs['dataset_cfg'].get('survival', False)
        self.weighted_samples = kwargs['dataset_cfg'].get('weighted_samples', False)

        self.load_data_module()

 

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        if not self.survival:
            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')
            self.test_dataset = self.instancialize(state='test')
        else:
            dataset = self.instancialize()
            split_csv_path = os.path.join(self.kwargs['dataset_cfg'].split_dir, 'splits_{}.csv'.format(self.kwargs['dataset_cfg'].fold))
            self.train_dataset, self.val_dataset, self.test_dataset = dataset.return_splits(from_id=False, csv_path=split_csv_path)


    def train_dataloader(self):
        if self.dataset_name in ['MIL_Graph_dataset', 'Survival_Graph_dataset']:
            from torch_geometric.loader import DataLoader
        else:
            from torch.utils.data import DataLoader
        if self.weighted_samples:
            weights = make_weights_for_balanced_classes_split(self.train_dataset)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, sampler=sampler)
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)

    def val_dataloader(self):
        if self.dataset_name in ['MIL_Graph_dataset', 'Survival_Graph_dataset']:
            from torch_geometric.loader import DataLoader
        else:
            from torch.utils.data import DataLoader
        if self.survival:
            return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        else:
            return [
                DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False), 
                DataLoader(self.test_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
            ]

    def test_dataloader(self):
        if self.dataset_name in ['MIL_Graph_dataset', 'Survival_Graph_dataset']:
            from torch_geometric.loader import DataLoader
        else:
            from torch.utils.data import DataLoader
        if self.survival:
            return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)
        else:
            return DataLoader(self.test_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)


    def load_data_module(self):
        # camel_data.py
        # CamelData(data.Dataset)
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name {camel_name}')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args: # args to init
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)