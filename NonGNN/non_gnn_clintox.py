import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdchem
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def get_molecule_features(mol):
    # Atom Features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feat = [
            atom.GetAtomicNum(),           
            int(atom.GetChiralTag() != rdchem.CHI_UNSPECIFIED), 
            atom.GetDegree(),              
            atom.GetFormalCharge(),       
            atom.GetTotalNumHs(),          
            atom.GetNumRadicalElectrons(), 
            int(atom.GetHybridization()),  
            int(atom.GetIsAromatic()),     
            int(atom.IsInRing()),          
        ]
        atom_features.append(atom_feat)
    
    atom_features = np.array(atom_features)
    atom_stats = {
        'atom_mean': np.mean(atom_features, axis=0),
        'atom_sum': np.sum(atom_features, axis=0),
        'atom_max': np.max(atom_features, axis=0),
        'atom_min': np.min(atom_features, axis=0),
    }

    # Bond Features
    if mol.GetNumBonds() > 0:
        bond_features = []
        for bond in mol.GetBonds():
            bond_feat = [
                int(bond.GetBondType()),    
                int(bond.IsInRing()),       
                int(bond.GetIsConjugated()),
                int(bond.GetStereo() != rdchem.BondStereo.STEREONONE), 
            ]
            bond_features.append(bond_feat)
        
        bond_features = np.array(bond_features)
        bond_stats = {
            'bond_mean': np.mean(bond_features, axis=0),
            'bond_sum': np.sum(bond_features, axis=0),
            'bond_max': np.max(bond_features, axis=0),
            'bond_min': np.min(bond_features, axis=0),
        }
    else:
        bond_stats = {
            'bond_mean': np.zeros(4),
            'bond_sum': np.zeros(4),
            'bond_max': np.zeros(4),
            'bond_min': np.zeros(4),
        }

    return atom_stats, bond_stats

def get_gnn_atom_features(mol):
    features = []
    
    for atom in mol.GetAtoms():
        atom_feat = [
            atom.GetAtomicNum(),           
            int(atom.GetChiralTag() != rdchem.CHI_UNSPECIFIED),  
            atom.GetDegree(),             
            atom.GetFormalCharge(),       
            atom.GetTotalNumHs(),         
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()), 
            int(atom.GetIsAromatic()),    
            int(atom.IsInRing()),         
        ]
        features.append(atom_feat)

    features = np.array(features)
    
    return {
        'mean': np.mean(features, axis=0),
        'sum': np.sum(features, axis=0),
        'max': np.max(features, axis=0),
        'min': np.min(features, axis=0),
    }

def molecule_to_features(smiles_list):
    feat_matrix = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print("Molecule not found")
        else:
            atom_stats, bond_stats = get_molecule_features(mol)
            feat_matrix.append(np.concatenate([
                atom_stats['atom_mean'],
                atom_stats['atom_sum'],
                bond_stats['bond_mean'],
                bond_stats['bond_sum']
            ]))
    return feat_matrix

def filter_broken_smiles_and_remap_folds(smiles_list, labels, folds):
    valid_idx = []
    filtered_smiles = []
    filtered_labels = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_idx.append(i)
            filtered_smiles.append(smi)
            filtered_labels.append(labels[i])

    old_to_new_index = {old: new for new, old in enumerate(valid_idx)}

    remapped_folds = []
    for fold in folds:
        remapped_fold = {
            "train_idx": [old_to_new_index[i] for i in fold["train_idx"] if i in old_to_new_index],
            "val_idx":   [old_to_new_index[i] for i in fold["val_idx"]   if i in old_to_new_index],
            "test_idx":  [old_to_new_index[i] for i in fold["test_idx"]  if i in old_to_new_index],
        }
        remapped_folds.append(remapped_fold)

    return pd.Series(filtered_smiles), pd.Series(filtered_labels), remapped_folds


# Model Training 
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_val_scores = model.predict_proba(X_val)[:, 1]
        y_test_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_val_scores = model.decision_function(X_val)
        y_test_scores = model.decision_function(X_test)
    else:
        raise ValueError("Model does not support probability or decision scores for AUC.")
    
    val_auc = roc_auc_score(y_val, y_val_scores)
    test_auc = roc_auc_score(y_test, y_test_scores)
    
    return val_auc, test_auc

# Hyperparameter Optimization
def run_fold(model_type, params, fold_data):
    train_idx = fold_data["train_idx"]
    val_idx = fold_data["val_idx"]
    test_idx = fold_data["test_idx"]

    X_train = X.iloc[train_idx,:]
    X_val = X.iloc[val_idx,:]
    X_test = X.iloc[test_idx,:]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    if model_type == 'RandomForest':
        model = RandomForestClassifier (
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=1729
        )
    elif model_type == 'XGBoost':
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            random_state=1729
        )
    elif model_type == 'Ridge':
        model = RidgeClassifier(alpha=params['alpha'])
    elif model_type == 'SVM':
        model = SVC(
            C=params['C'],
            gamma=params['gamma'],
            probability=True
        )

    val_auc, test_auc = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    return val_auc, test_auc 

def objective(params):
    
    print(f"Trying: {params}")

    fold_val_auc = []
    fold_test_auc = []

    for fold in folds:
        val_auc, test_auc  = run_fold(
            model_type = model_name,
            params = params,
            fold_data= fold
        )
        fold_val_auc.append(val_auc)
        fold_test_auc.append(test_auc)

    avg_val_auc = np.mean(fold_val_auc)
    avg_test_auc = np.mean(fold_test_auc)

    return {'loss': -avg_val_auc, 'status': STATUS_OK, 'test_aucs' : fold_test_auc}


model_name = ""

if __name__ == '__main__':
    dataset = pd.read_csv('clintox.csv')
    smiles_list = dataset['smiles']
    y = dataset['CT_TOX']

    print(smiles_list)
    print(y)

    def load_split(json_path):
        with open(json_path, 'r') as f:
            split = json.load(f)
        return split

    folds = load_split('clintox_split_5fold.json')

    smiles_list, y, folds = filter_broken_smiles_and_remap_folds(smiles_list, y, folds)

    X = pd.DataFrame(molecule_to_features(smiles_list))
    print(X.head())

    # Model selection
    search_space = {
        'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_samples_split' : hp.quniform('min_samples_split', 2, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'alpha': hp.loguniform('alpha', np.log(0.1), np.log(10)),
        'C': hp.loguniform('C', np.log(0.1), np.log(10)),
        'gamma': hp.loguniform('gamma', np.log(0.001), np.log(0.1)),
    }


    for m in ['Ridge', 'SVM', 'XGBoost', 'RandomForest']:
        model_name = m
        print(f"Training {model_name}...")
        trials = Trials()

        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=25, 
            trials=trials
        )
        print("Best parameters for: ", m, ":" , best)

        # Retrieve best trial
        best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
        print(f"Validation Loss (best): {best_trial['loss']:.4f}")
        print("Test AUC (corresponding):")
        print("Values:", [f"{rmse:.4f}" for rmse in best_trial['test_aucs']])
        print(f"Mean: {np.mean(best_trial['test_aucs']):.4f}")
        print(f"Std dev: {np.std(best_trial['test_aucs']):.4f}")



