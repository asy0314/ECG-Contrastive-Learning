import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class EcgDataset(Dataset):
    def __init__(self, hdf5_file, labeled_only=False, label_type='subclass', include_folds=None, exclude_folds=None, transform=None):
        self.hdf5_file_path = hdf5_file
        self.hf = None  # Placeholder for file handle in each worker
        self.indices = None  # actual indices of data
        self.labeled_only = labeled_only
        assert label_type in {'superclass', 'subclass', 'binary'}
        self.label_type = label_type
        if label_type == 'superclass':
            self.num_classes = 5
        elif label_type == 'subclass':
            self.num_classes = 23
        elif label_type == 'binary':
            self.num_classes = 2

        assert not ((include_folds is not None) and (exclude_folds is not None))
        if include_folds is None:
            include_folds = []
        if exclude_folds is None:
            exclude_folds = []
        self.include_folds = include_folds
        self.exclude_folds = exclude_folds

        self.transform = transform

    def _ensure_file_open(self):
        # Open a new file handle if not already open
        if self.hf is None:
            # it is UNNECESSARY to set swmr=True for READ-ONLY access
            self.hf = h5py.File(self.hdf5_file_path, 'r')
            strat_fold = np.asarray(self.hf['strat_fold'])
            
            folds_used = set(np.unique(strat_fold).tolist())
            include_folds = set(self.include_folds)
            if len(include_folds) > 0:
                folds_used = folds_used.intersection(include_folds)
            exclude_folds = set(self.exclude_folds)
            folds_used = list(folds_used.difference(exclude_folds))
            if self.labeled_only:
                is_labeled = np.asarray(self.hf['is_labeled'])
            else:
                is_labeled = np.ones_like(strat_fold, dtype=bool)
            
            valid_samples = np.isin(strat_fold, folds_used)
            valid_samples = np.logical_and(valid_samples, is_labeled)
            self.indices = np.nonzero(valid_samples)[0]

    def __len__(self):
        self._ensure_file_open()
        return len(self.indices)

    
    def __getitem__(self, idx):
        self._ensure_file_open()

        # get the actual data index
        actual_idx = self.indices[idx]

        # retrieve data from HDF5 dataset
        ecg_sample = self.hf['ecg_data'][actual_idx]
        ecg_sample = torch.from_numpy(ecg_sample).float()

        if self.label_type == 'binary':
            multilabel = self.hf[f'label_superclass'][actual_idx]
            # Check if the first element is 1 and all other elements are 0, then negate the result
            target = not ((multilabel[0] > 0) and np.all(multilabel[1:] == 0))
            target = torch.tensor(target, dtype=int)
        else:
            target = self.hf[f'label_{self.label_type}'][actual_idx]
            target = torch.from_numpy(target).float()

        # print('ecg_sample')
        # print(ecg_sample)
        # print(ecg_sample.size())

        # Apply the transform, if specified
        if self.transform:
            ecg_sample = self.transform(ecg_sample)
        
        return ecg_sample, target

    def __del__(self):
        # Ensure each worker closes its own file handle
        if self.hf is not None:
            self.hf.close()
    