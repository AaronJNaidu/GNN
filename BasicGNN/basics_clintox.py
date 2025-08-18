# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, AttentiveFP

from torch.utils.data import Subset
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import numpy as np
import copy

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time
import json

from collections import defaultdict
import random

import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def build_model(model_name, in_dim, hidden_dim, num_layers, output_dim):
    if model_name == 'GCN':
        convs = [GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
    elif model_name == 'GAT':
        convs = []
        heads = 4
        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim * heads
            if i == num_layers - 1:
                convs.append(GATConv(in_channels, hidden_dim, heads=1, concat=False))
            else:
                convs.append(GATConv(in_channels, hidden_dim, heads=heads))
    elif model_name == 'GraphSAGE':
        convs = [SAGEConv(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
    elif model_name == 'GIN':
        convs = []
        for i in range(num_layers):
            if i == 0:
                nn1 = torch.nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            else:
                nn1 = torch.nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            convs.append(GINConv(nn1))
    elif model_name == 'AttentiveFP':
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = AttentiveFP(in_channels=in_dim, hidden_channels=hidden_dim,
                                         out_channels=output_dim, edge_dim=dataset[0].edge_attr.shape[1],
                                         num_layers=num_layers, dropout=0.1, num_timesteps=2)

            def forward(self, x, edge_index, edge_attr, batch):
                x = x.float()
                return self.model(x, edge_index, edge_attr, batch)
        return Net()
    else:
        raise ValueError("Unknown model")

    # For all non-AttentiveFP models
    class Net(torch.nn.Module):
        def __init__(self, conv_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList(conv_layers)
            self.lin = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x, edge_index, batch):
            x = x.float()
            edge_index = edge_index.to(torch.long)
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.1, training=self.training)
            x = global_mean_pool(x, batch)
            return self.lin(x)

    return Net(convs)

def train_and_evaluate(model_name, model, optimizer, train_loader, val_loader, test_loader, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            batch.x = batch.x.float()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
                loss = F.binary_cross_entropy_with_logits(pred, batch.y)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print("Epoch", epoch, ": val loss", val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
            test_preds.append(torch.sigmoid(pred).cpu().numpy())
            test_targets.append(batch.y.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)

    model.eval()
    val_preds = []
    val_targets = []
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
            val_preds.append(torch.sigmoid(pred).cpu().numpy())
            val_targets.append(batch.y.cpu().numpy())

    val_preds = np.concatenate(test_preds)
    val_targets = np.concatenate(test_targets)



    try:
        val_roc_auc = roc_auc_score(val_targets, val_preds, average = 'macro')
        test_roc_auc = roc_auc_score(test_targets, test_preds, average='macro')
    except ValueError:
        val_roc_auc = float('nan')
        test_roc_auc = float('nan')

    print(f"Val Loss: {best_val_loss:.4f}")
    print(f"Val ROC-AUC: {val_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")

    return best_val_loss, val_roc_auc, test_roc_auc

# Update run_fold to use output_dim
def run_fold(model_name, in_dim, hidden_dim, num_layers, lr, weight_decay, fold_data, output_dim):
    train_idx = fold_data["train_idx"]
    val_idx = fold_data["val_idx"]
    test_idx = fold_data["test_idx"]
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    model = build_model(model_name, in_dim, hidden_dim, num_layers, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_and_evaluate(model_name, model, optimizer, train_loader, val_loader, test_loader, num_epochs=50, patience=10)

# Define objective
def objective(params):
    hidden_dim = params['hidden_dim']
    num_layers = params['num_layers']
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    print(f"Trying: {params}")
    fold_val_loss = []
    fold_test_auc = []
    fold_val_auc = []

    for fold in folds:
        val_loss, val_auc, test_auc = run_fold(
            model_name=model_name,
            in_dim=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            fold_data=fold,
            output_dim=dataset.num_classes
        )
        fold_val_loss.append(val_loss)
        fold_test_auc.append(test_auc)
        fold_val_auc.append(val_auc)

    avg_val_loss = np.mean(fold_val_loss)
    avg_test_auc = np.mean(fold_test_auc)
    avg_val_auc = np.mean(fold_val_auc)

    return {'loss': avg_val_loss, 'status': STATUS_OK, 'test_aucs': fold_test_auc, 'val_auc': avg_val_auc}

# Prepare dataset
dataset = MoleculeNet(root='data', name='Clintox')
smiles_list = [data.smiles for data in dataset]

def load_split(json_path):
    with open(json_path, 'r') as f:
        split = json.load(f)
    return split

folds = load_split('esol_split_5fold.json')

# Hyperopt space
hidden_dims = [32, 64, 128, 256]
num_layers_list = [1, 2, 3, 4, 5]
num_timesteps_list = [1, 2, 3, 4]
param_choices = {
    'hidden_dim': hidden_dims,
    'num_layers': num_layers_list,
    'num_timesteps' : num_timesteps_list
}
def get_mapped_param(trial_val, param_name):
    if param_name in param_choices:
        return param_choices[param_name][trial_val]
    return trial_val

search_space = {
    'hidden_dim': hp.choice('hidden_dim', hidden_dims),
    'num_layers': hp.choice('num_layers', num_layers_list),
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'num_timesteps': hp.choice('num_timesteps', num_timesteps_list)
}

# Train all models
for m in ['AttentiveFP' ,'GCN', 'GAT', 'GraphSAGE', 'GIN']:
    model_name = m
    print(f"\nTraining {model_name} on Clintox...")
    trials = Trials()

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials
    )

    print(f"Best parameters for {model_name}: {best}")
    best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
    print(f"Validation Loss (best): {best_trial['loss']:.4f}")
    print(f"Validation AUC (best): {best_trial['val_auc']:.4f}")
    print("Test ROC-AUC (corresponding):")
    print("Values:", [f"{auc:.4f}" for auc in best_trial['test_aucs']])
    print(f"Mean: {np.mean(best_trial['test_aucs']):.4f}")
    print(f"Std dev: {np.std(best_trial['test_aucs']):.4f}")




    # Plot
    param_names = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']
    results = []
    for t in trials.trials:
        if t['result']['status'] != 'ok':
            continue
        result = t['result']
        params = t['misc']['vals']
        row = {'test_auc': np.mean(result.get('test_aucs'))}
        for p in param_names:
            val = params.get(p, [None])[0]
            row[p] = get_mapped_param(val, p)
        results.append(row)

    fig, axes = plt.subplots(1, len(param_names), figsize=(5 * len(param_names), 4))
    for i, param in enumerate(param_names):
        values = [r[param] for r in results]
        test_aucs = [r['test_auc'] for r in results]
        axes[i].scatter(values, test_aucs, alpha=0.7, c='green')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Test ROC-AUC')
        axes[i].set_title(f'{param} vs Test ROC-AUC')
        axes[i].grid(True)

    plt.tight_layout()
    plot_filename = f"test_auc_vs_hyperparams_{m}_clintox.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
