import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdchem
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def get_molecule_features(mol):
    """Compute both atom-level and bond-level features for a molecule.
    Returns:
        Dictionary with aggregated atom and bond features.
    """
    # --- Atom Features ---
    atom_features = []
    for atom in mol.GetAtoms():
        atom_feat = [
            atom.GetAtomicNum(),           # Atomic number
            int(atom.GetChiralTag() != rdchem.CHI_UNSPECIFIED),  # Chirality (binary)
            atom.GetDegree(),              # Degree
            atom.GetFormalCharge(),       # Formal charge
            atom.GetTotalNumHs(),          # Number of hydrogens
            atom.GetNumRadicalElectrons(), # Radical electrons
            int(atom.GetHybridization()),  # Hybridization (as integer)
            int(atom.GetIsAromatic()),     # Is aromatic (binary)
            int(atom.IsInRing()),          # Is in ring (binary)
        ]
        atom_features.append(atom_feat)
    
    # Aggregate atom features
    atom_features = np.array(atom_features)
    atom_stats = {
        'atom_mean': np.mean(atom_features, axis=0),
        'atom_sum': np.sum(atom_features, axis=0),
        'atom_max': np.max(atom_features, axis=0),
        'atom_min': np.min(atom_features, axis=0),
    }

    # --- Bond Features ---
    if mol.GetNumBonds() > 0:
        bond_features = []
        for bond in mol.GetBonds():
            bond_feat = [
                int(bond.GetBondType()),    # Bond type (1=SINGLE, 2=DOUBLE, etc.)
                int(bond.IsInRing()),       # Is in ring (binary)
                int(bond.GetIsConjugated()),# Is conjugated (binary)
                int(bond.GetStereo() != rdchem.BondStereo.STEREONONE),  # Stereochemistry (binary)
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
        # Handle molecules with no bonds (e.g., single-atom)
        bond_stats = {
            'bond_mean': np.zeros(4),
            'bond_sum': np.zeros(4),
            'bond_max': np.zeros(4),
            'bond_min': np.zeros(4),
        }

    return atom_stats, bond_stats

def get_gnn_atom_features(mol):
    """Convert atom features into molecule-level statistics"""
    features = []
    
    for atom in mol.GetAtoms():
        atom_feat = [
            atom.GetAtomicNum(),           # Atomic number
            int(atom.GetChiralTag() != rdchem.CHI_UNSPECIFIED),  # Chirality (binary)
            atom.GetDegree(),             # Degree
            atom.GetFormalCharge(),       # Formal charge
            atom.GetTotalNumHs(),         # Number of hydrogens
            atom.GetNumRadicalElectrons(),# Radical electrons
            int(atom.GetHybridization()), # Hybridization (as integer)
            int(atom.GetIsAromatic()),    # Is aromatic (binary)
            int(atom.IsInRing()),         # Is in ring (binary)
        ]
        features.append(atom_feat)

    # Convert to numpy array (num_atoms x 9)
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
            return None
        
        atom_stats, bond_stats = get_molecule_features(mol)
        feat_matrix.append(np.concatenate([
            atom_stats['atom_mean'],
            atom_stats['atom_sum'],
            bond_stats['bond_mean'],
            bond_stats['bond_sum']
        ]))
    return feat_matrix

# ==== Model Training ====
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    return val_rmse, test_rmse, test_r2

# ==== Hyperparameter Optimization ====
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
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=1729
        )
    elif model_type == 'XGBoost':
        model = XGBRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            random_state=1729
        )
    elif model_type == 'Ridge':
        model = Ridge(alpha=params['alpha'])
    elif model_type == 'SVR':
        model = SVR(
            C=params['C'],
            gamma=params['gamma'],
        )


    # Train and evaluate
    val_rmse, test_rmse, test_r2 = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    return test_rmse, test_r2, val_rmse 

def objective(params):
    
    print(f"Trying: {params}")

    fold_val_rmse = []
    fold_test_rmse = []

    for fold in folds:
        test_rmse, _, val_rmse  = run_fold(
            model_type = model_name,
            params = params,
            fold_data= fold
        )
        fold_val_rmse.append(val_rmse)
        fold_test_rmse.append(test_rmse)

    avg_val_rmse = np.mean(fold_val_rmse)
    avg_test_rmse = np.mean(fold_test_rmse)

    return {'loss': avg_val_rmse, 'status': STATUS_OK, 'test_rmses' : fold_test_rmse}


model_name = ""

if __name__ == '__main__':

    # Load ESOL dataset
    dataset = pd.read_csv('esol.csv')
    smiles_list = dataset['smiles']
    y = dataset['ESOL predicted log solubility in mols per litre']

    def load_split(json_path):
        with open(json_path, 'r') as f:
            split = json.load(f)
        return split

    folds = load_split('esol_split_5fold.json')

    # Compute features
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


    for m in ['Ridge', 'SVR', 'XGBoost', 'RandomForest']:
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
        print("Test RMSE (corresponding):")
        print("Values:", [f"{rmse:.4f}" for rmse in best_trial['test_rmses']])
        print(f"Mean: {np.mean(best_trial['test_rmses']):.4f}")
        print(f"Std dev: {np.std(best_trial['test_rmses']):.4f}")

