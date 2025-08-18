import numpy as np
import dgl
import torch
import argparse
from models.model import HGNN
from models.utils import GraphDataset_Classification,GraphDataLoader_Classification,\
                  AUC,RMSE,\
                  GraphDataset_Regression,GraphDataLoader_Regression
from torch.optim import Adam
from data.split_data import get_classification_dataset,get_regression_dataset
from collections import defaultdict

import torch
import json
from pathlib import Path

#Clintox
'''
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

#ESOL
'''
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

    
def main(args, graph_cache):
    # Update args with suggested hyperparameters
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

    args.atom_layer_type = params['atom_layer_type']
    args.motif_layer_type = params['motif_layer_type']

    best_test_score = 0

    for fold_id in range(args.folds):
        print('fold: ',fold_id)


        if task_type=='classification':
            metric=AUC
            train_gs, train_ls, train_tw = graph_cache[fold_id]['train']
            val_gs, val_ls = graph_cache[fold_id]['val']
            test_gs, test_ls = graph_cache[fold_id]['test']

            all_train_gs = train_gs + val_gs
            all_train_ls = torch.cat([train_ls, val_ls], dim=0)

            print(len(all_train_ls), len(test_ls))

            pos_counts = all_train_ls.sum(dim=0)
            neg_counts = all_train_ls.shape[0] - pos_counts
            merged_pos_weight = neg_counts / (pos_counts + 1e-8)

            train_ds = GraphDataset_Classification(all_train_gs, all_train_ls)
            train_dl = GraphDataLoader_Classification(train_ds, num_workers=0, batch_size=args.batch_size,
                                       shuffle=args.shuffle)
            criterion_atom = torch.nn.BCEWithLogitsLoss(pos_weight=merged_pos_weight.to(args.device))
            criterion_fg = torch.nn.BCEWithLogitsLoss(pos_weight=merged_pos_weight.to(args.device))
        else:
            metric=RMSE
            best_test_score = 10
            train_gs, train_ls = graph_cache[fold_id]['train']
            val_gs, val_ls = graph_cache[fold_id]['val']
            test_gs, test_ls = graph_cache[fold_id]['test']

            all_train_gs = train_gs + val_gs
            all_train_ls = torch.cat([train_ls, val_ls], dim=0)

            print(len(all_train_ls), len(test_ls))
            train_ds = GraphDataset_Regression(all_train_gs, all_train_ls)
            train_dl = GraphDataLoader_Regression(train_ds, num_workers=0, batch_size=args.batch_size,
                                       shuffle=args.shuffle)
            criterion_atom =torch.nn.MSELoss(reduction='none')
            criterion_fg =torch.nn.MSELoss(reduction='none')
            
        dist_loss=torch.nn.MSELoss(reduction='none')
        
        test_gs=dgl.batch(test_gs).to(args.device)
        test_labels=test_ls.to(args.device)
        train_labels = all_train_ls.to(args.device)

        model = HGNN(train_labels.shape[1],
                      args,
                      criterion_atom,
                      criterion_fg,
                      ).to(args.device)
        print(model)
        opt = Adam(model.parameters(),lr=args.learning_rate)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,eta_min=1e-4,verbose=True)
        
        for epoch in range(args.epoch):
            print("Epoch: ", epoch)
            model.train()
            traYAll = []
            traPredictAll = []
            for i, (gs, labels) in enumerate(train_dl):
                traYAll += labels.detach().cpu().numpy().tolist()
                gs = gs.to(args.device)
                labels = labels.to(args.device).float()
                af=gs.nodes['atom'].data['feat']
                bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
                fnf = gs.nodes['func_group'].data['feat']
                fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                molf=gs.nodes['molecule'].data['feat']
                atom_pred,fg_pred= model(gs, af, bf,fnf,fef,molf,labels)
                ##############################################
                if task_type=='classification':
                    logits=(torch.sigmoid(atom_pred)+torch.sigmoid(fg_pred))/2
                    dist_atom_fg_loss=dist_loss(torch.sigmoid(atom_pred),torch.sigmoid(fg_pred)).mean()
                else:
                    logits=(atom_pred+fg_pred)/2
                    dist_atom_fg_loss=dist_loss(atom_pred,fg_pred).mean()
                loss_atom=criterion_atom(atom_pred,labels).mean()
                loss_motif=criterion_fg(fg_pred,labels).mean()
                loss=loss_motif+loss_atom+args.dist*dist_atom_fg_loss
                ##################################################
                opt.zero_grad()
                loss.backward()
                opt.step()
                traPredictAll += logits.detach().cpu().numpy().tolist()  
            train_score,train_AUPRC=metric(traYAll,traPredictAll)
            model.eval()
            with torch.no_grad():
                    test_af = test_gs.nodes['atom'].data['feat']
                    test_bf = test_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    test_fnf = test_gs.nodes['func_group'].data['feat']
                    test_fef=test_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    test_molf=test_gs.nodes['molecule'].data['feat']
                    test_logits_atom,test_logits_motif= model(test_gs, test_af, test_bf, test_fnf,test_fef,test_molf,test_labels)
                    ###################################################
                    if task_type=='classification':
                        test_logits=(torch.sigmoid(test_logits_atom)+torch.sigmoid(test_logits_motif))/2
                    else:
                        test_logits=(test_logits_atom+test_logits_motif)/2                
                    test_score,test_AUPRC=metric(test_labels.detach().cpu().numpy().tolist(), test_logits.detach().cpu().numpy().tolist())
                    ###################################################
                    if task_type=='classification':
                        print('#####################')
                        print("-------------------Epoch {}-------------------".format(epoch))
                        print("Train AUROC: {}".format(train_score)," Train AUPRC: {}".format(train_AUPRC))
                        print("Test AUROC: {}".format(test_score)," Test AUPRC: {}".format(test_AUPRC))
                        if test_score > best_test_score:
                            best_test_score = test_score
                            print("Saving model")
                            torch.save(model.state_dict(), "gnn_new_clintox.pth")

                    elif task_type=='regression':
                        print('#####################')
                        print("-------------------Epoch {}-------------------".format(epoch))
                        print("Train RMSE: {}".format(train_score))
                        print('Test RMSE: {}'.format(test_score))
                        if test_score < best_test_score:
                            best_test_score = test_score
                            print("Saving model")
                            torch.save(model.state_dict(), "gnn_new_esol.pth")
       
    print("AUROC/RMSE:\n")
    print(best_test_score)
    return best_test_score
    
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str,choices=['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE','ESOL', 'FreeSolv', 'Lipophilicity'],
                   default='BBBP', help='dataset name')
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
    p.add_argument('--folds',type=int,default=10,help='k folds validation')
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
    p.add_argument('--ScaleBlock',type=str,choices=['Share','Contextual'],default='Contextual',help='Self-Rescaling Block'
                   )
    p.add_argument('--heads',type=int,default=4,help='Multi-head num')

    p.add_argument('--atom_layer_type',type=str,choices=['nnconv', 'nnconvmean', 'nnconvmax','gcn', 'gat', 'gin', 'sage'],default='nnconv',help='Atom MPNN layer type')
    p.add_argument('--motif_layer_type',type=str,choices=['nnconv', 'nnconvmean', 'nnconvmax' ,'gcn', 'gat', 'gin', 'sage'],default='nnconv',help='Motif MPNN layer type')


    args = p.parse_args()

    max_score_list=[]
    max_aupr_list=[]
    task_type=None
    if args.dataset == 'ClinTox':
        task_type='classification'
    else:
        task_type='regression'

    graph_cache = defaultdict(dict)

    for fold_id in range(args.folds):
        if task_type == 'classification':
            train_gs, train_ls, train_tw, val_gs, val_ls, test_gs, test_ls = get_classification_dataset(
                args.dataset, args.n_jobs, fold_id
            )
            graph_cache[fold_id]['train'] = (train_gs, train_ls, train_tw)
            graph_cache[fold_id]['val'] = (val_gs, val_ls)
            graph_cache[fold_id]['test'] = (test_gs, test_ls)
        else: 
            train_gs, train_ls, val_gs, val_ls, test_gs, test_ls = get_regression_dataset(
                args.dataset, args.n_jobs, fold_id
            )
            graph_cache[fold_id]['train'] = (train_gs, train_ls)
            graph_cache[fold_id]['val'] = (val_gs, val_ls)
            graph_cache[fold_id]['test'] = (test_gs, test_ls)

    main(args, graph_cache)
