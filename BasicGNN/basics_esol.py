# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, AttentiveFP
from torch.utils.data import Subset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import copy
import gc

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time

from collections import defaultdict
import random

import math
import matplotlib.pyplot as plt
import json

device = torch.device('cpu')


# Define the model-building function
def build_model(model_name, in_dim, hidden_dim, num_layers, output_dim = 1):
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
                edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 1
                self.model = AttentiveFP(in_channels=in_dim, hidden_channels=hidden_dim,
                                         out_channels=output_dim, edge_dim=edge_dim,
                                         num_layers=num_layers, dropout=0.1, num_timesteps = 2)

            def forward(self, x, edge_index, batch, edge_attr=None):
                return self.model(x.float(), edge_index, edge_attr, batch).squeeze(-1)

        return Net()
    else:
        raise ValueError("Unknown model")

    # Wrap conv layers in a model
    class Net(torch.nn.Module):
        def __init__(self, conv_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList(conv_layers)
            self.lin = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index, batch):
            x = x.float()
            edge_index = edge_index.to(torch.long)

            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.1, training=self.training)
            x = global_mean_pool(x, batch)
            return self.lin(x).squeeze(-1)

    return Net(convs)

# Training loop with early stopping and evaluation
def train_and_evaluate(model_name, model, optimizer,  train_loader, val_loader, test_loader, num_epochs=50, patience=10,):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            # Convert data types if needed
            batch.x = batch.x.float()
            batch.edge_index = batch.edge_index.long()

            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out.flatten(), batch.y.squeeze())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        for batch in val_loader:
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
                val_losses.append(F.mse_loss(pred.flatten(), batch.y.squeeze()).item())
        val_loss = np.mean(val_losses)
        print("Epoch", epoch, ": val loss", math.sqrt(val_loss))
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)

    # Evaluation on the test set
    model.eval()
    test_preds = []
    test_targets = []
    for batch in test_loader:
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr) if model_name == 'AttentiveFP' else model(batch.x, batch.edge_index, batch.batch)
            test_preds.append(pred.flatten().cpu().numpy())
            test_targets.append(batch.y.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)

    mse = mean_squared_error(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)

    print(f"Val RMSE: {math.sqrt(best_val_loss):.4f}")
    print(f"Test RMSE: {math.sqrt(mse):.4f}")
    print(f"Test R2: {r2:.4f}")

    return math.sqrt(mse), r2, math.sqrt(best_val_loss)

def run_fold(model_name, in_dim, hidden_dim, num_layers, lr, weight_decay, fold_data):
    train_idx = fold_data["train_idx"]
    val_idx = fold_data["val_idx"]
    test_idx = fold_data["test_idx"]
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)


    model = build_model(model_name, in_dim, hidden_dim, num_layers, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_and_evaluate(model_name, model, optimizer,
                                train_loader, val_loader, test_loader, num_epochs=50, patience=10)


hidden_dims = [32, 64, 128, 256]
num_layers_list = [1, 2, 3, 4, 5]

param_choices = {
    'hidden_dim': hidden_dims,
    'num_layers': num_layers_list,
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
}

model_name = ""

def objective(params):
    hidden_dim = params['hidden_dim']
    num_layers = params['num_layers']
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    print(f"Trying: {params}")

    fold_val_rmse = []
    fold_test_rmse = []

    for fold in folds:
        test_rmse, _, val_rmse  = run_fold(
            model_name=model_name,
            in_dim=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            fold_data=fold
        )
        fold_val_rmse.append(val_rmse)
        fold_test_rmse.append(test_rmse)

    avg_val_rmse = np.mean(fold_val_rmse)
    avg_test_rmse = np.mean(fold_test_rmse)

    torch.cuda.empty_cache()
    gc.collect()

    return {'loss': avg_val_rmse, 'status': STATUS_OK, 'test_rmses' : fold_test_rmse}

dataset = MoleculeNet(root='data', name='ESOL')
smiles_list = [data.smiles for data in dataset]


def load_split(json_path):
    with open(json_path, 'r') as f:
        split = json.load(f)
    return split

folds = load_split('esol_split_5fold.json')

for m in ['AttentiveFP', 'GCN', 'GAT', 'GraphSAGE', 'GIN']:
    model_name = m
    print(f"Training {model_name}...")
    trials = Trials()

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=30, 
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

    param_names = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']
    results = []

    for t in trials.trials:
        if t['result']['status'] != 'ok':
            continue
        result = t['result']
        params = t['misc']['vals']
        
        row = {'test_loss': np.mean(result.get('test_rmses'))}
        for p in param_names:
            val = params.get(p, [None])[0]
            row[p] = get_mapped_param(val, p)
        results.append(row)

    # Plot test loss vs each param
    fig, axes = plt.subplots(1, len(param_names), figsize=(5 * len(param_names), 4))

    for i, param in enumerate(param_names):
        values = [r[param] for r in results]
        test_losses = [np.mean(r['test_loss']) for r in results]

        axes[i].scatter(values, test_losses, alpha=0.7, c='darkorange')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Test Loss')
        axes[i].set_title(f'{param} vs Test Loss')
        axes[i].grid(True)

        plt.tight_layout()
        plot_filename = f"test_rmse_vs_hyperparams_{m}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

