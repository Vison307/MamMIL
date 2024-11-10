from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
import addict


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self, dataset_cfg):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        
        # ----> Extract dataset_cfg
        csv_path = dataset_cfg.get('csv_path')
        mode = dataset_cfg.get('mode', 'path')
        apply_sig = dataset_cfg.get('apply_sig', False)
        shuffle = dataset_cfg.get('shuffle', False) 
        seed = dataset_cfg.get('seed', 7)
        print_info = dataset_cfg.get('print_info', True)
        n_bins = dataset_cfg.get('n_bins', 4)
        ignore= dataset_cfg.get('ignore', [])
        patient_strat = dataset_cfg.get('patient_strat', False)
        label_col = dataset_cfg.get('label_col', None)
        filter_dict = dataset_cfg.get('filter_dict', {})
        eps = dataset_cfg.get('eps', 1e-6)
        
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)


        slide_data = pd.read_csv(csv_path, low_memory=False)
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        import pdb
        #pdb.set_trace()

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            # print(f'Patient: {patient} has {slide_ids} slides')
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, mode = self.mode, metadata= self.apply_sig, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, metadata = self.apply_sig, mode = self.mode, data_dir=self.data_dir, num_classes=self.num_classes)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, metadata = self.apply_sig, mode = self.mode, data_dir=self.data_dir, num_classes=self.num_classes)

            else:
                test_split = None
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            # print(all_splits.head())
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1) 
        df.to_csv(filename, index = False)

class SurvivalGraphDataset(Generic_WSI_Survival_Dataset):
    def __init__(self, dataset_cfg):
        super(SurvivalGraphDataset, self).__init__(dataset_cfg)
        self.data_dir = dataset_cfg.get('data_dir')
        self.mode = 'path'
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = float(self.slide_data['censorship'][idx])
        slide_ids = self.patient_dict[case_id]
        # print(slide_ids)

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if self.data_dir: # NOTE: currently enter this
            if self.mode == 'path': # NOTE: currently only supports path mode
                path_features = []
                for slide_id in slide_ids:
                    # wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id))
                    wsi_bag = torch.load(wsi_path)
                    path_features.append(wsi_bag)
                # path_features = torch.cat(path_features, dim=0)
                assert len(path_features) == 1
                path_features = path_features[0]
                return {'data': path_features, 'label': label, 'event_time': event_time, 'c': c}
                # return (path_features, torch.zeros((1,1)), label, event_time, c)
            elif self.mode == 'cluster':
                path_features = []
                cluster_ids = []
                for slide_id in slide_ids:
                    wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_bag = torch.load(wsi_path)
                    path_features.append(wsi_bag)
                    cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt']) #! no fname2ids?
                path_features = torch.cat(path_features, dim=0)
                cluster_ids = torch.Tensor(cluster_ids)
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (path_features, cluster_ids, genomic_features, label, event_time, c)

            elif self.mode == 'omic':
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (torch.zeros((1,1)), genomic_features, label, event_time, c)

            elif self.mode == 'pathomic':
                path_features = []
                for slide_id in slide_ids:
                    wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_bag = torch.load(wsi_path)
                    path_features.append(wsi_bag)
                path_features = torch.cat(path_features, dim=0)
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (path_features, genomic_features, label, event_time, c)

            elif self.mode == 'coattn':
                path_features = []
                for slide_id in slide_ids:
                    wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_bag = torch.load(wsi_path)
                    path_features.append(wsi_bag)
                path_features = torch.cat(path_features, dim=0)
                omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

            else:
                raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            ### <--
        else:
            return slide_ids, label, event_time, c


class Generic_Split(SurvivalGraphDataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--

    def pre_loading(self, thread=8):
        # set flag
        self.cache_flag = True

        ids = list(range(len(self)))
        from multiprocessing.pool import ThreadPool
        exe = ThreadPool(thread)
        exe.map(self.__getitem__, ids)



if __name__ == '__main__':
    dataset_cfg = {
        'csv_path': 'dataset_csv/LUAD_processed.csv',
        'data_dir': 'data/TCGA_FEATURES/TCGA_LUAD_512_at_level0/pt_files',
        'shuffle': False,
        'seed': 41,
        'patient_strat': False,
        'n_bins': 4,
        'label_col': 'survival_months',
        'split_dir': 'splits/TCGA_LUAD_survival_kfold',
    }
    dataset = SurvivalDataset(dataset_cfg)
    train, val, test = dataset.return_splits(from_id=False, csv_path=os.path.join(dataset_cfg['split_dir'], 'splits_0.csv'))
    print(len(train), len(val))
    print(train[0])