from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import pickle
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    if len(dataset)>train_size+valid_size+test_size:
        test_size += len(dataset)-(train_size+valid_size+test_size)
    if return_test:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
        return train_loader, val_loader

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    batch_o_indices = []
    o_and_nbr_indices = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, rel_indices), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        o_and_nbr_indices.append(torch.LongTensor(np.array(rel_indices)+base_idx))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)

        base_idx += n_i
    # return O atom neighbours
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx, o_and_nbr_indices),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of pickle files using a specified num_nbrs and o_atom dist as in the original cgcnn. 

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, random_seed=123):
        self.root_dir = root_dir
        assert os.path.exists(root_dir), 'root_dir of pickles does not exist!'

    def __len__(self):
        len_atom_feas_pickles = len(os.listdir(os.path.join(self.root_dir, "atom_feas")))
        assert len_atom_feas_pickles==len(os.listdir(os.path.join(self.root_dir, "nbr_feas")))
        assert len_atom_feas_pickles==len(os.listdir(os.path.join(self.root_dir, "nbr_fea_indices")))
        assert len_atom_feas_pickles==len(os.listdir(os.path.join(self.root_dir, "rel_indices")))
        return len_atom_feas_pickles

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        atom_fea = pickle.load(open(os.path.join(self.root_dir, "atom_feas/atom_fea_{}.p".format(idx)), "rb"))
        nbr_fea = pickle.load(open(os.path.join(self.root_dir, "nbr_feas/nbr_fea_{}.p".format(idx)), "rb"))
        nbr_fea_idx = pickle.load(open(os.path.join(self.root_dir, "nbr_fea_indices/nbr_fea_idx_{}.p".format(idx)), "rb"))
        rel_indices = pickle.load(open(os.path.join(self.root_dir, "rel_indices/rel_index_{}.p".format(idx)), "rb"))
        target = pickle.load(open(os.path.join(self.root_dir, "targets/target_{}.p".format(idx)), "rb"))
        cif_id = pickle.load(open(os.path.join(self.root_dir, "cif_ids/cif_id_{}.p".format(idx)), "rb"))
        #print(idx, cif_id, target)
        
        return (atom_fea, nbr_fea, nbr_fea_idx, rel_indices), target, cif_id
