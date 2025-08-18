import torch, argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from torch.nn import BCELoss
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.SaltRemover import SaltRemover
import hyperopt
from hyperopt import fmin, hp, Trials
from hyperopt.early_stop import no_progress_loss
from io import BytesIO
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
import datetime
import argparse
import numpy as np
import dgl
import torch
import csv
import os
from models.model import HGNN
from models.utils import GraphDataset_Classification,GraphDataLoader_Classification,\
                  AUC,RMSE,\
                  GraphDataset_Regression,GraphDataLoader_Regression
from torch.optim import Adam
from data.split_data import get_classification_dataset,get_regression_dataset
from collections import defaultdict
from dgl.nn.pytorch.explain import GNNExplainer
import cairosvg


warnings.filterwarnings("ignore")
remover = SaltRemover()
bad = ['He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']
_use_shared_memory = True
torch.backends.cudnn.benchmark = True

cmap = plt.cm.bwr 

def map_to_rgb(value):
    rgba = cmap(value)
    return (rgba[0], rgba[1], rgba[2])

def visualize_mol_with_annotations_contribution(smiles, scores, prediction=None, actual=None, name=None, title="Contribution Visualization"):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    if len(scores) != num_atoms:
        raise ValueError(f"Number of scores ({len(scores)}) != number of atoms ({num_atoms})")

    # Normalize scores to [0,1]
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = cm.get_cmap("viridis")  # or "cool", "plasma", etc.
    norm_scores = 2 * (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) - 1
    atom_colors = {i: cmap(norm(float(score))) for i, score in enumerate(norm_scores)}

    # Draw molecule with attention and FG highlights
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.drawOptions().addAtomIndices = False

    highlight_atoms = list(atom_colors.keys())
    highlight_colors = atom_colors.copy()

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,
                                            highlightAtoms=highlight_atoms,
                                            highlightAtomColors=highlight_colors
    )
    drawer.FinishDrawing()
    mol_img = drawer.GetDrawingText()
    mol_img = plt.imread(BytesIO(mol_img), format='png')

    # Plot with matplotlib
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(mol_img)
    ax.axis('off')
    header = title
    if name:
        header += f"\n: {name}"
    plt.suptitle(title, fontsize=14, weight="bold", y=0.92)
    plt.title(smiles, fontsize=9, y=0.94)

    # Add annotation below image
    annotation = ""
    if prediction is not None and actual is not None:
        annotation = f"Prediction: {prediction:.3f} | Actual: {actual:.3f}"
        ax.text(0.5, -0.08, annotation, transform=ax.transAxes,
                ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white', edgecolor='gray'))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Normalised contribution", fontsize=12)

    # Save to bytes
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def visualize_mol_with_annotations(smiles, scores, prediction=None, actual=None, name=None, title="Attention Visualization"):
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    if len(scores) != num_atoms:
        raise ValueError(f"Number of scores ({len(scores)}) != number of atoms ({num_atoms})")

    # Normalize scores to [0,1]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("cool")  # or "viridis", "plasma", etc.
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)
    atom_colors = {i: cmap(norm(float(score))) for i, score in enumerate(norm_scores)}

    # Identify functional groups (e.g., OH, NH2, COOH)
    fg_atoms = []
    patt_list = {
        "OH": Chem.MolFromSmarts("[OX2H]"),
        "NH2": Chem.MolFromSmarts("[NX3;H2]"),
        "COOH": Chem.MolFromSmarts("C(=O)[OH]")
    }
    for label, patt in patt_list.items():
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            fg_atoms.extend(match)

    # Draw molecule with attention and FG highlights
    drawer = Draw.MolDraw2DCairo(500, 500)
    drawer.drawOptions().addAtomIndices = False



    highlight_atoms = list(atom_colors.keys())
    highlight_colors = atom_colors.copy()
    #for idx in fg_atoms:
    #    highlight_colors[idx] = (1.0, 1.0, 0.0)  # Yellow for functional groups

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,
                                            highlightAtoms=highlight_atoms,
                                            highlightAtomColors=highlight_colors
    )
    drawer.FinishDrawing()
    mol_img = drawer.GetDrawingText()
    mol_img = plt.imread(BytesIO(mol_img), format='png')

    # Plot with matplotlib
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(mol_img)
    ax.axis('off')
    header = title
    if name:
        header += f"\n: {name}"
    plt.suptitle(title, fontsize=14, weight="bold", y=0.92)
    plt.title(smiles, fontsize=9, y=0.94)

    # Add annotation below image
    annotation = ""
    if prediction is not None and actual is not None:
        annotation = f"Prediction: {prediction:.3f} | Actual: {actual:.3f}"
        ax.text(0.5, -0.08, annotation, transform=ax.transAxes,
                ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white', edgecolor='gray'))

    # Add colorbar
    cmap = plt.cm.cool
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Attention Score", fontsize=12)

    # Save to bytes
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def get_molecule_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown Molecule"
    return Chem.MolToSmiles(mol)  # or fallback to canonical SMILES

def compute_integrated_gradients(model, g, target_idx=0, baseline=None, steps=50):
    model.eval()
    device = next(model.parameters()).device
    g = g.to(device)

    # Extract the node features for 'atom' nodes
    input_x = g.nodes['atom'].data['feat'].detach()

    if baseline is None:
        baseline = torch.zeros_like(input_x).to(device)

    integrated_grads = torch.zeros_like(input_x).to(device)

    # You also need to extract all other features your model requires
    bf = g.edges[('atom', 'interacts', 'atom')].data['feat']
    fnf = g.nodes['func_group'].data['feat']
    fef = g.edges[('func_group', 'interacts', 'func_group')].data['feat']
    molf = g.nodes['molecule'].data['feat']

    for alpha in torch.linspace(0, 1, steps).to(device):
        interpolated_x = (baseline + alpha * (input_x - baseline)).detach()
        interpolated_x.requires_grad_(True)

        # Temporarily replace atom features in graph with interpolated_x
        g.nodes['atom'].data['feat'] = interpolated_x

        # Forward pass: pass all required features explicitly
        atom_pred, fg_pred = model(g, interpolated_x, bf, fnf, fef, molf, None)  # assuming labels=None for IG

        # Combine outputs as in your training code
        output = (torch.sigmoid(atom_pred) + torch.sigmoid(fg_pred)) / 2
        squeezed_output = output.squeeze()

        if squeezed_output.dim() == 0:
            pred = squeezed_output
        else:
            pred = squeezed_output[target_idx]

        model.zero_grad()
        pred.backward()

        if interpolated_x.grad is not None:
            grads = interpolated_x.grad.detach()
        else:
            grads = torch.zeros_like(interpolated_x)

        integrated_grads += grads

        # Clear gradients for next iteration
        interpolated_x.grad.zero_()

    avg_grads = integrated_grads / steps
    attributions = (input_x - baseline) * avg_grads

    #print("attributions shape:", attributions.shape)  # should be [num_atoms, num_features]
    node_attributions = attributions.sum(dim=1)       # sum over features to get attribution per atom
    return node_attributions.cpu()


class HGNNWrapper(nn.Module):
    def __init__(self, model, bf, fnf, fef, mf):
        super(HGNNWrapper, self).__init__()
        self.model = model
        # Store fixed features (bond features, full node features, etc.)
        self.bf = bf
        self.fnf = fnf
        self.fef = fef
        self.mf = mf

    def forward(self, *args, **kwargs):
        # g: DGLGraph
        # af: node feature tensor from explainer (shape: [num_nodes, feat_dim])
        # Combine node features from explainer with stored other features
        # Your original model expects (g, af, bf, fnf, fef, mf)
        g = kwargs.get('graph', None)
        af = kwargs.get('feat', None)

        if g is None or af is None:
            raise ValueError("HGNNWrapper.forward expects 'graph' and 'feat' keyword arguments")

        # Make sure stored features are on the same device as af
        device = af.device
        bf = self.bf.to(device)
        fnf = self.fnf.to(device)
        fef = self.fef.to(device)
        mf = self.mf.to(device)

        atom_output, motif_output = self.model(g, af, bf, fnf, fef, mf, labels=None)
        # Return outputs as tuple (matching your model output)
        #return atom_output, motif_output
        combined = (torch.sigmoid(atom_output) + torch.sigmoid(motif_output)) / 2
        return combined  # shape [1, num_classes]
    
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

def explain_and_visualize_rdkit(model, g, af, bf, fnf, fef, mf, smiles, device="cpu", pred="",label="", return_masks = False):
    # Move data to device
    g = g.to(device)
    #print(g.etypes)
    af, bf, fnf, fef, mf = af.to(device), bf.to(device), fnf.to(device), fef.to(device), mf.to(device)

    # Wrap HGNN
    wrapped_model = HGNNWrapper(model, bf, fnf, fef, mf).to(device)
    wrapped_model.eval()

    # GNNExplainer
    explainer = GNNExplainer(
        wrapped_model,
        num_hops=3,
        lr=0.01,
        num_epochs=200
    )

    #feat_tuple = (af, bf, fnf, fef, mf)
    node_features = af
    node_feat_mask, edge_mask = explainer.explain_graph(g, node_features)

    # Convert to numpy
    node_mask = node_feat_mask.detach().cpu().numpy()
    edge_mask = edge_mask.detach().cpu().numpy()

    # Normalize for coloring
    node_mask = (node_mask - node_mask.min()) / (node_mask.max() - node_mask.min() + 1e-6)
    edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-6)

    svg = None
    '''
    # RDKit mol
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Map DGL edges to RDKit bonds
    edge_list = g.edges(etype = ('atom', 'interacts', 'atom'))
    edge_tuples = list(zip(edge_list[0].cpu().numpy(), edge_list[1].cpu().numpy()))

    # Atom colors
    atom_colors = {i: map_to_rgb(node_mask[i]) for i in range(len(node_mask))}

    # Bond colors
    bond_colors = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for k, (u, v) in enumerate(edge_tuples):
            if (u == a1 and v == a2) or (u == a2 and v == a1):
                bond_colors[bond.GetIdx()] = map_to_rgb(edge_mask[k])
                break

    # Draw molecule
    drawer = Draw.MolDraw2DSVG(500, 300)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    valid_atoms = [i for i in range(num_atoms) if i in atom_colors]
    valid_bonds = [b for b in range(num_bonds) if b in bond_colors]

    drawer.DrawMolecule(
        mol,
        highlightAtoms=valid_atoms,
        highlightAtomColors={k: atom_colors[k] for k in valid_atoms},
        highlightBonds=valid_bonds,
        highlightBondColors={k: bond_colors[k] for k in valid_bonds}
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    save_with_colorbar(png_bytes, node_mask, edge_mask, save_dir / f"mol_{i}_gnnexplainer.png", smiles, pred, label)

    print("Visualization saved to explanation.svg")
    '''
    if return_masks:
        return node_mask, edge_mask
    return svg

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def save_with_colorbar(png_bytes, node_mask, edge_mask, save_path, smiles, pred_value, actual_value):
    fig, ax = plt.subplots(figsize=(6,6))

    # Show molecule image
    image = Image.open(BytesIO(png_bytes))
    ax.imshow(image)
    ax.axis('off')

    # Add heading at top
    plt.text(0.5, 1.12, "GNN Explainer", transform=ax.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Truncate SMILES to first 30 chars + '...' if longer
    short_smiles = smiles if len(smiles) <= 30 else smiles[:30] + "..."

    # Add SMILES below the heading, centered
    plt.text(0.5, 1.05, f"SMILES: {short_smiles}", transform=ax.transAxes, ha='center', va='bottom', fontsize=10)

    # Add colorbar on the right
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.bwr
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance (low → high)')

    # Add predicted and actual values below the image
    plt.text(0.5, -0.05, f"Predicted: {pred_value:.3f}   Actual: {actual_value:.3f}",
             transform=ax.transAxes, ha='center', va='top', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str,choices=['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE','ESOL', 'FreeSolv', 'Lipophilicity'],
                   default='BBBP', help='dataset name')
    p.add_argument('--pretrain', type=str, default="", help = "Path to pretrained model params")
    p.add_argument('--seed', type=int, default=0, help='seed used to shuffle dataset')
    p.add_argument('--atom_in_dim', type=int, default=37, help='atom feature init dim')
    p.add_argument('--bond_in_dim', type=int, default=13, help='bond feature init dim')
    p.add_argument('--ss_node_in_dim', type=int, default=50, help='func group node feature init dim')
    p.add_argument('--ss_edge_in_dim', type=int, default=37, help='func group edge feature init dim')
    p.add_argument('--mol_in_dim', type=int, default=167, help='molecule fingerprint init dim')
    p.add_argument('--learning_rate', type=float, default=5e-3, help='Adam learning rate')
    p.add_argument('--epoch', type=int, default=50, help='train epochs')
    p.add_argument('--batch_size', type=int, default=200, help='batch size for train dataset')
    p.add_argument('--num_neurons', type=list, default=[512],help='num_neurons in MLP')
    p.add_argument('--input_norm', type=str, default='layer', help='input norm')
    p.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate in MLP')
    p.add_argument('--hid_dim', type=int, default=96, help='node, edge, fg hidden dims in Net')
    p.add_argument('--device', type=str, default='cuda:0', help='fitting device')
    p.add_argument('--dist',type=float,default=0.005,help='dist loss func hyperparameter lambda')
    p.add_argument('--split_ratio',type=list,default=[0.8,0.1,0.1],help='ratio to split dataset')
    p.add_argument('--folds',type=int,default=5,help='k folds validation')
    p.add_argument('--n_jobs',type=int,default=10,help='num of threads for the handle of the dataset')
    p.add_argument('--resdual',type=bool,default=False,help='resdual choice in message passing')
    p.add_argument('--shuffle',type=bool,default=False,help='whether to shuffle the train dataset')
    p.add_argument('--attention',type=bool,default=True,help='whether to use global attention pooling')
    p.add_argument('--step',type=int,default=4,help='message passing steps')
    p.add_argument('--agg_op',type=str,choices=['max','mean','sum'],default='mean',help='aggregations in local augmentation')
    p.add_argument('--mol_FP',type=str,choices=['atom','ss','both','none'],default='ss',help='cat mol FingerPrint to Motif or Atom representation'
                   )
    p.add_argument('--gating_func',type=str,choices=['Softmax','Sigmoid','Identity'],default='Sigmoid',help='Gating Activation Function'
                   )
    p.add_argument('--ScaleBlock',type=str,choices=['Norm','Contextual'],default='Contextual',help='Self-Rescaling Block'
                   )
    p.add_argument('--heads',type=int,default=4,help='Multi-head num')
    p.add_argument('--max_eval',type=int,default=50,help='Number of hyperparameter combinations')

    args = p.parse_args()

    if args.dataset == 'ClinTox':
        task_type = 'classification'
    else:
        task_type = 'regression'

    fold_id = 0
    graph_cache = defaultdict(dict)
    
    if task_type == 'classification':
        train_gs, train_ls, train_tw, val_gs, val_ls, test_gs, test_ls = get_classification_dataset(
            args.dataset, args.n_jobs, fold_id
        )
        graph_cache[fold_id]['train'] = (train_gs, train_ls, train_tw)
        graph_cache[fold_id]['val'] = (val_gs, val_ls)
        graph_cache[fold_id]['test'] = (test_gs, test_ls)
        params = {
            'batch_size': 32,
            'hid_dim': 128,
            'drop_rate': 0.2,
            'learning_rate': 0.001, 
            'heads': 2,
            'dist' : 0.2,

            'gating_func' : 'Sigmoid',
            'agg_op' : 'max',
            'num_neurons' : [256],
            'step' : 3,

            'atom_layer_type' : 'gat',
            'motif_layer_type' : 'nnconv'
            }
    else: 
        train_gs, train_ls, val_gs, val_ls, test_gs, test_ls = get_regression_dataset(
            args.dataset, args.n_jobs, fold_id
        )
        graph_cache[fold_id]['train'] = (train_gs, train_ls)
        graph_cache[fold_id]['val'] = (val_gs, val_ls)
        graph_cache[fold_id]['test'] = (test_gs, test_ls)
        params = {
            'batch_size': 64,
            'hid_dim': 96,
            'drop_rate': 0.1,
            'learning_rate': 0.0005, 
            'heads': 4,
            'dist' : 0.02,

            'gating_func' : 'Identity',
            'agg_op' : 'max',
            'num_neurons' : [256, 256],
            'step' : 5,

            'atom_layer_type' : 'gat',
            'motif_layer_type' : 'nnconv'
            }

    args.learning_rate = params['learning_rate']
    args.drop_rate = params['drop_rate']
    args.batch_size = params['batch_size']
    args.hid_dim = params['hid_dim']
    args.dist = params['dist']
    args.heads = params['heads']

    args.gating_func = params['gating_func']
    args.agg_op = params['agg_op']
    args.num_neurons = list(params['num_neurons'])
    args.step = params['step']

    args.gating_func = params['gating_func']
    args.agg_op = params['agg_op']

    args.atom_layer_type = params['atom_layer_type']
    args.motif_layer_type = params['motif_layer_type']

    test_score_list = []
    val_score_list = []

    if task_type=='classification':
        metric=AUC
        train_gs, train_ls, train_tw = graph_cache[fold_id]['train']
        val_gs, val_ls = graph_cache[fold_id]['val']
        test_gs, test_ls = graph_cache[fold_id]['test']

        print(len(train_ls),len(val_ls),len(test_ls),train_tw)
        train_ds = GraphDataset_Classification(train_gs, train_ls)
        train_dl = GraphDataLoader_Classification(train_ds, num_workers=0, batch_size=args.batch_size,
                                    shuffle=args.shuffle)
        task_pos_weights=train_tw
        criterion_atom = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
        criterion_fg = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
    else:
        metric=RMSE
        train_gs, train_ls = graph_cache[fold_id]['train']
        val_gs, val_ls = graph_cache[fold_id]['val']
        test_gs, test_ls = graph_cache[fold_id]['test']

        print(len(train_ls),len(val_ls),len(test_ls))
        train_ds = GraphDataset_Regression(train_gs, train_ls)
        train_dl = GraphDataLoader_Regression(train_ds, num_workers=0, batch_size=args.batch_size,
                                    shuffle=args.shuffle)
        criterion_atom =torch.nn.MSELoss(reduction='none')
        criterion_fg =torch.nn.MSELoss(reduction='none')
        
    dist_loss=torch.nn.MSELoss(reduction='none')

    #val_gs = dgl.batch(val_gs).to(args.device)
    #val_labels=val_ls.to(args.device)

    #test_gs=dgl.batch(test_gs).to(args.device)
    #test_labels=test_ls.to(args.device)

    model = HGNN(2,
                    args,
                    criterion_atom,
                    criterion_fg,
                    ).to(args.device)


    state_dict = torch.load(args.pretrain)
    model.load_state_dict(state_dict)
    model.eval()

    from pathlib import Path

    # Directory to save visualizations
    save_dir = Path("visualizations_explainer_0815")
    save_dir.mkdir(exist_ok=True)

    explainer = GNNExplainer(
    model,
    num_hops=3,          # should match your message-passing depth
    lr=0.01,
    num_epochs=200
)

    #print(len(test_gs))
    results_list = []
    # Loop through a few test molecules
    for i, g in enumerate(test_gs):
        smiles = g.smiles  # make sure your dataset graphs store SMILES here
        #print(f"Label tensor shape at index {i}: {test_ls[i].shape}")
        #print(test_ls[i])
        print(i)
        label = test_ls[i][1].item()
        mol = Chem.MolFromSmiles(smiles)
        atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

        # === 1. Forward pass ===
        with torch.no_grad():
            af = g.nodes['atom'].data['feat']
            bf = g.edges[('atom', 'interacts', 'atom')].data['feat']
            fnf = g.nodes['func_group'].data['feat']
            fef = g.edges[('func_group', 'interacts', 'func_group')].data['feat']
            mf = g.nodes['molecule'].data['feat']

            atom_output, motif_output = model(g.to(args.device), af, bf, fnf, fef, mf, labels = None)
            test_logits=(torch.sigmoid(atom_output)+torch.sigmoid(motif_output))/2
            test_logits = test_logits.squeeze()
            prob_1 = test_logits[1]/(test_logits[0] + test_logits[1])
            #print(prob_1)
            pred = prob_1.squeeze().item()

        # === 2. Get atom-level scores ===
        # Option A: Use integrated gradients
        ig_scores = compute_integrated_gradients(model, g, target_idx=0, steps=50)
        ig_scores = 2 * (ig_scores - ig_scores.min()) / (ig_scores.max() - ig_scores.min() + 1e-9) - 1
        # Option B: If your model outputs attention weights directly:
        # scores = g.ndata["attention_weights"].cpu().numpy()

        # === 3. Visualization ===
        '''
        img_bytes = visualize_mol_with_annotations_contribution(
            smiles, scores,
            prediction=pred, actual=label,
            name=f"mol_{i}", title="Atom Contribution"
        )

        # === 4. Save image ===
        img_path = save_dir / f"mol_{i}.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        results_list.append((smiles, pred, label))

        '''
        # === 3. GNNExplainer ===
        node_mask, svg_output = explain_and_visualize_rdkit(
            model,
            g, af, bf, fnf, fef, mf,
            smiles=smiles,  # example SMILES
            device= args.device,
            pred=pred, label=label,
            return_masks = True
        )

        expl_scores = node_mask

        for atom_idx, (sym, ig_val, expl_val) in enumerate(zip(atom_symbols, ig_scores, expl_scores)):
            results_list.append({
                "Model": "Original",
                "Molecule": smiles,
                "MoleculeIndex" : i,
                "AtomIndex": atom_idx,
                "AtomSymbol": sym,
                "Contribution": ig_val.item(),
                "ExplainerImportance": expl_val
            })

    #print(f"Saved visualizations to {save_dir}")
    '''
    filename = "clintox_orig_preds.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["SMILES", "Prediction", "Actual"])
        for row in results_list:
            writer.writerow(row)
        '''
    
    with open("atom_importances_new.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)