import torch, json
from functools import partial
import dgl.backend as F
import dgllife.data as dgldata
from dgllife.utils import RandomSplitter,ScaffoldSplitter
from .MolGraph_Construction import smiles_to_Molgraph,ATOM_FEATURIZER, BOND_FEATURIZER
import numpy as np
from random import Random
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from dgl.data.utils import  Subset
try:
    from rdkit import Chem
except ImportError:
    pass
def count_and_log(message, i, total, log_every_n):
    if (log_every_n is not None) and ((i + 1) % log_every_n == 0):
        print('{} {:d}/{:d}'.format(message, i + 1, total))
def prepare_mols(dataset, mols, sanitize, log_every_n=1000):
    if mols is not None:
        # Sanity check
        assert len(mols) == len(dataset), \
            'Expect mols to be of the same size as that of the dataset, ' \
            'got {:d} and {:d}'.format(len(mols), len(dataset))
    else:
        if log_every_n is not None:
            print('Start initializing RDKit molecule instances...')
        mols = []
        for i, s in enumerate(dataset.smiles):
            count_and_log('Creating RDKit molecule instance',
                          i, len(dataset.smiles), log_every_n)
            mols.append(Chem.MolFromSmiles(s, sanitize=sanitize))

    return mols
def scaffold_split(dataset, frac=None, balanced=True, include_chirality=False, ramdom_state=0):
    if frac is None:
        frac = [0.8, 0.1, 0.1]
    assert sum(frac) == 1
    mol_list = prepare_mols(dataset, None, True)
    n_total_valid = int(np.floor(frac[1] * len(mol_list)))
    n_total_test = int(np.floor(frac[2] * len(mol_list)))
    n_total_train = len(mol_list) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, mol in enumerate(mol_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    random = Random(ramdom_state)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    return [Subset(dataset, train_index),
                Subset(dataset, valid_index),
                Subset(dataset, test_index)]

def load_split(json_path):
    with open(json_path, 'r') as f:
        split = json.load(f)
    return split

def predetermined_split(dataset, fold, name):
    if name == "ClinTox":
        folds = load_split('clintox_split_smiles.json')
    if name == "ESOL":
        folds = load_split('esol_split_smiles.json')

    this_fold = folds[fold]
    train_smiles = this_fold['train_smiles']
    valid_smiles = this_fold['val_smiles']
    test_smiles = this_fold['test_smiles']

    train_index, valid_index, test_index = [], [], []

    for i, s in enumerate(dataset.smiles):
            if s in train_smiles:
                train_index.append(i)
            elif s in valid_smiles:
                valid_index.append(i)
            elif s in test_smiles:
                test_index.append(i)
            else:
                print("Smiles: ", s, " not found")

    return [Subset(dataset, train_index),
                Subset(dataset, valid_index),
                Subset(dataset, test_index)]

def get_classification_dataset(dataset: str,
                               n_jobs: int,
                               fold_id: int):
    assert dataset in ['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE']

    def get_task_pos_weights(labels, masks):
        num_pos = F.sum(labels, dim=0)
        num_indices = F.sum(masks, dim=0)
        task_pos_weights = (num_indices - num_pos) / num_pos
        return task_pos_weights

    def get_data(sub_data):
        gs, ys, ms = [], [], []
        for i in range(len(sub_data)):
            smiles = sub_data[i][0]
            g = sub_data[i][1]
            g.smiles = smiles
            gs.append(g)
            ys.append(sub_data[i][2])
            ms.append(sub_data[i][3])
        ys = torch.stack(ys)
        ms = torch.stack(ms)
        task_weights = get_task_pos_weights(ys, ms)
        return gs, ys, ms, task_weights

    mol_g = partial(smiles_to_Molgraph)
    data = getattr(dgldata, dataset)(mol_g,
                                     ATOM_FEATURIZER,
                                     BOND_FEATURIZER,
                                     n_jobs=n_jobs)
    
    """
    train, val,test= ScaffoldSplitter.train_val_test_split(dataset=data,sanitize=False,
                                                      frac_train=split_ratio[0], 
                                                      frac_val=split_ratio[1],
                                                      frac_test=split_ratio[2],scaffold_func='smiles')
    """
    train,val,test= predetermined_split(dataset=data, fold = fold_id, name = "ClinTox")
    train_gs, train_ls, train_masks, train_tw = get_data(train)
    val_gs, val_ls, val_masks, val_tw = get_data(val)
    test_gs, test_ls, test_masks, test_tw = get_data(test)
    return train_gs, train_ls,train_tw, val_gs, val_ls,test_gs,test_ls

def get_regression_dataset(dataset: str,
                           n_jobs: int,
                           fold_id: int):

    assert dataset in ['ESOL', 'FreeSolv', 'Lipophilicity']

    def get_datareg(sub_data):
        gs, ys=[], []
        for i in range(len(sub_data)):
            smiles = sub_data[i][0]
            g = sub_data[i][1]
            g.smiles = smiles
            gs.append(g)
            ys.append(sub_data[i][2])
        ys = torch.stack(ys)
        return gs, ys

    mol_g = partial(smiles_to_Molgraph)

    data = getattr(dgldata, dataset)(mol_g,
                                     ATOM_FEATURIZER,
                                     BOND_FEATURIZER,
                                     n_jobs=n_jobs)
    """
    train, val,test= RandomSplitter.train_val_test_split(dataset=data,sanitize=False,
                                                      frac_train=split_ratio[0],
                                                      frac_val=split_ratio[1],
                                                      frac_test=split_ratio[2],scaffold_func='smiles'
                                                      )
    """
    train,val,test= predetermined_split(dataset=data,fold=fold_id, name = "ESOL")
    train_gs, train_labels = get_datareg(train)
    val_gs, val_labels = get_datareg(val)
    test_gs, test_labels = get_datareg(test)
    return train_gs,train_labels,val_gs,val_labels,test_gs,test_labels