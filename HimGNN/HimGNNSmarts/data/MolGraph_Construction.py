from functools import partial
import dgl
from rdkit.Chem import MACCSkeys
import networkx as nx
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
import pandas as pd
import numpy as np
from .Substructure_Extraction import get_substructures
from dgllife.utils import BaseAtomFeaturizer, BaseBondFeaturizer
from dgllife.utils.featurizers import (
    atom_chirality_type_one_hot,
                                       atom_explicit_valence_one_hot, 
                                       atom_hybridization_one_hot,
                                       atom_is_aromatic_one_hot,
                                       atom_is_chiral_center,
                                       atom_is_in_ring_one_hot,
                                       atom_total_num_H_one_hot,
                                       atom_type_one_hot,
                                       atomic_number,
                                       bond_is_conjugated_one_hot,
                                       bond_is_in_ring,
                                       bond_type_one_hot,
                                       bond_stereo_one_hot
                                       )
ATOM_FEATURIZER = BaseAtomFeaturizer({'atom_type': partial(atom_type_one_hot,
                                                           allowable_set=["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"]),
                                      'atomic_number':atomic_number,
                                      'atom_explicit': atom_explicit_valence_one_hot,
                                      'atom_num_H': atom_total_num_H_one_hot,
                                      'atom_hybridization': atom_hybridization_one_hot,
                                      'aromatic': atom_is_aromatic_one_hot,
                                      'atom_in_ring': atom_is_in_ring_one_hot,
                                      'atom_chirality': atom_chirality_type_one_hot,
                                      'atom_chiral_center': atom_is_chiral_center})

BOND_FEATURIZER = BaseBondFeaturizer({'bond_type': bond_type_one_hot,
                                      'in_ring': bond_is_in_ring,
                                      'conj_bond': bond_is_conjugated_one_hot,
                                      'bond_stereo':bond_stereo_one_hot})

NODE_ATTRS = ['atom_type',
              'atomic_number',
              'atom_explicit',
              'atom_num_H',
              'atom_hybridization',
              'aromatic',
              'atom_in_ring',
              'atom_chirality',
              'atom_chiral_center']
EDGE_ATTRS = ['bond_type',
              'in_ring',
              'conj_bond',
              'bond_stereo']

TOXIC_SMARTS = "string"

'''
TOXIC_SMARTS = {"Anilines" : "c1ccccc1[NX3]([#1,CX4!R,$(c1ccccc1),$([CX3](=[OX1])[#6]),$([CX3](=[OX1])[OX2][#6]),$([SX4](=[OX1])(=[OX1])[#1,#6])])[#1,CX4!R,$(c1ccccc1),$([CX3](=[OX1])[#6]),$([CX3](=[OX1])[OX2][#6]),$([SX4](=[OX1])(=[OX1])[#1,#6])]",
                "Hydrazine" : "[#6,#1][NX3H1][NX3H1][#6,#1]",
                "Nitrobenzenes" :"a[$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[O-])]",
                "Dibenzazepines" :"N([CX4])([CX4])C1=Nc2ccccc2[NX3H1]c3c1cccc3",
                "Benzylamines" :"[#1,CX4!R][NX3]([#1,CX4!R])[CH2]c1ccccc1",
                "Propargyl amines" :" [#1,CX4,$(c1ccccc1)][CX2]#[CX2][NX3]([#1,CX4,$(c1ccccc1)])[#1,CX4,$(c1ccccc1)]",
                "Cyclopropyl amines" :"[CX4]1[CX4][CX4]1[NX3]([#1,CX4,$(c1ccccc1)])[#1,CX4,$(c1ccccc1)]",
                "N-Substituted-4-aryl-1,2,3,6-tetrahydropyridines" :"[NX3]1([CX4!R])[CX4][CX3]=[CX3](c2c([#1,#6,F,Cl,Br,I])c([#1,#6,F,Cl,Br,I])c([#1,#6,F,Cl,Br,I])c([#1,#6,F,Cl,Br,I])c2([#1,#6,F,Cl,Br,I]))[CX4][CX4]1",
                "N-Substituted-4-arylpiperidin-4-ol" :"[NX3]1([CX4!R])[CX4][CX4][CX4]([OH1])(c2c([#1,#6,F,Cl,Br,I])c([#1,#6,F,Cl,Br,I])c([#1,#6,F,Cl,Br,I])c([#1,#6,F,Cl,Br,I])c2([#1,#6,F,Cl,Br,I]))[CX4][CX4]1",
                "Formamides" :"[CX3H1](=[OX1])[NX3H1][#6]",
                "Hydantoins (glycolylurea)" :"[NX3H1]1[CX3](=[OX1])[CX4]([#1,#6])([#1,#6])[NX3H1][CX3]1(=[OX1])",
                "Thioamides" :"[#1,#6][CX3](=[SX1])[NX3]([#1,#6])[#1,#6]",
                "Thioureas" : "[#1,#6][NX3]([#1,#6])[CX3](=[SX1])[#7]",
                "Sulfonylureas" : "[#6][Sv6X4](=[OX1])(=[OX1])[NX3H1][CX3](=[OX1])[NX3H1][#6]",
                "Thiols" : "[SX2!H0][#1,#6&!$([CX3]([SH1])=[OX1,SX1,NX2])]",
                "Disulfides" : "[#6,#1][SX2][SX2][#6,#1]",
                "parahydroquinones" : "[$(c1([OH1])c([OH1])cccc1),$(c1([OH1])ccc([OH1])cc1)]",
                "paraquinones" : "[$([#6]1(=[OX1])-,:[#6]=,:[#6]-,:[#6](=[OX1])-,:[#6]=,:[#6]1),$([#6]1(=[OX1])-,:[#6](=[OX1])-,:[#6]=,:[#6]-,:[#6]=,:[#6]1)]",
                "paraalkylphenols" : "[$(c1([OH1])c([CX4]([#1,#6])[#1,#6])cccc1),$(c1([OH1])ccc([CX4]([#1,#6])[#1,#6])cc1)]",
                "Quinone methide" : "[$([OX1]=[#6X3]1-,:[#6X3]=,:[#6X3]-,:[#6X3]=,:[#6X3]-,:[#6X3]1=[CX3]([#1,#6])[#1,#6]),$([OX1]=[#6]1-,:[#6]=,:[#6]-,:[#6](=[CX3]([#1,#6])[#1,#6])-,:[#6]=,:[#6]1)]",
                "Benzo-dioxolanes" : "c1ccc([OX2]2)c([OX2][CX4H1]2[OH1])c1",
                "3-Methylene indoles" : "c1ccc([nX3]2([#1,#6]))c(c([CH2][#1,#6])c2)c1",
                "Furans" : "c1ccc[oX2]1",
                "Thiophenes" : "c1ccc[sX2]1",
                "Thiazoles" : "[$(c1([#1,CX4!R])c([#1,CX4!R])[nX2]c([#1,CX4!R])[sX2]1),$(c1([#1,CX4!R])c([#1,CX4!R])[nX2]c([NX3]([#1,CX4!R])[#1,CX4!R])[sX2]1)]",
                "Thiazolidinediones" : "[SX2]1[CX3](=[OX1])[NX3H1][CX3](=[OX1])[CX4]1[#6]",
                #"Arenes" : "NOT [!#6!#1] AND c1ccccc1",
                #"Bromoarenes" : "NOT [!#6!#1!Br] AND c1ccccc1Br",
                "4-Halopyridines" : "[$(n1c([Cl,Br,I,F])cccc1),$(n1cccc([Cl,Br,I,F])cc1)]",
                "Heterocycles " : "nc[$([OX2][Sv6X4](=[OX1])(=[OX1])[OH]),$([OX2][Sv6X4](=[OX1])(=[OX1])[OX2][CH3]),$([OX2][Sv6X4](=[OX1])(=[OX1])[CH3]),$([OX2][Sv6X4](=[OX1])(=[OX1])[CF3]),$([OX2][Sv6X4](=[OX1])(=[OX1])c1ccc([CH3])cc1),$([I,Br,Cl])]",
                "Alkynes" : "[#6][CX2]#[CX2][#1,#6]",
                "Michael acceptors" : "[$([CX3]=[CX3][$([$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[O-])]),$([CX3](=[OX1])[OX2][#1,#6]),$([Sv6X4](=[OX1])(=[OX1])[OH]),$([CX3](=[OX1])[#1,F,Cl,Br,I]),$([CX2]#[NX1]),$([CX4]([F,Cl])([F,Cl])[F,Cl]),$([CH2][OH]),$([CH2][$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[O-])]),$([CH2][F,Cl])]),$([CX3]=[CX3][CX3](=[OX1])[#1,#6])]",
                "Alkylhalides" : "[$([CX4]([F,Cl,Br,I])([H,!F!Cl!Br!I])([H,!F!Cl!Br!I])[H,!F!Cl!Br!I]),$([CX4]([#6])([F,Cl,Br,I])([F,Cl,Br,I])[H,!F!Cl!Br!I])]",
                "Carboxylic acids" : "[#6][CX3](=[OX1])[OH1]",
                "Fluoroethyl ethers and amines" : "[$([F][CX4][$([CH2]),$([CH]),$(C[OH1])][OX2][#1,#6]),$([F][CX4][$([CH2]),$([CH]),$(C[OH1])][NX3][#6])]"
                }
'''

def get_mechanism_flags(mol):
    return {name: int(mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))) for name, smarts in TOXIC_SMARTS.items()}

def get_global_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    flags = get_mechanism_flags(mol)

    return [
        sum(flags.values()), max(flags.values())
    ]

def SSgraph_edge_construction(smiles,f2a_edges,num_funcs):
        """construct the edge between substructures&fix substructures-atoms transformation"""
        #input:smiles seq of mol
        #      substructures number 
        #      the corresponding atoms of substurctures
        #output:substructures connection(edge inf)
        #       atoms-edge transformation
        mol = Chem.MolFromSmiles(smiles)
        edge=[]
        edge_atoms={}
        processing_SS=[]
        for i in range(num_funcs):
            processing_SS.append(list((f2a_edges[1][np.argwhere(f2a_edges[0].numpy()==i).ravel()]).numpy()))
        
        """delete a substructure if all of its atoms are contained in another substructure"""
        del_index=[]
        for i in processing_SS:
            for j in processing_SS:
                if i==j:
                    continue
                if(set(i).intersection(j)==set(i)):
                    del_index.append(i)
        for i in del_index:
            try:
                processing_SS.remove(i)
            except:
                pass
        num_funcs=len(processing_SS)
        #update corresponding relations between atoms and substurctures
        tensor_atoms=[]
        tensor_SS=[]
        for i in processing_SS:
            tensor_atoms.extend(list(i))
            tensor_SS.extend([processing_SS.index(i) for j in range(len(i))])
        tensor_atoms=torch.tensor(tensor_atoms)
        tensor_SS=torch.tensor(tensor_SS)
        a2f_edges=(tensor_atoms,tensor_SS)
        f2a_edges=(tensor_SS,tensor_atoms)
        
        #constructure a edge if two substructures have share atoms
        for i,SS1 in enumerate(processing_SS):
            for j,SS2 in enumerate(processing_SS):
                if i==j:
                    continue
                if(len(set(SS1)&set(SS2))>0) and [i,j] not in edge:
                    edge.append([i,j])
                    
        atom_idx=[]
        for e in edge:
            edge_atoms.update({str(e):list(set(processing_SS[e[0]])&set(processing_SS[e[1]]))})
        
        #constructure a edge if two substructures do not have share atoms
        for atom in mol.GetAtoms():
            for neighbor in atom.GetNeighbors():
                if [atom.GetIdx(),neighbor.GetIdx()] not in atom_idx and [neighbor.GetIdx(),atom.GetIdx()] not in atom_idx:
                    atom_idx.append([atom.GetIdx(),neighbor.GetIdx()])
        left_idx=-1
        right_idx=-1
        
        for atom1,atom2 in atom_idx:
            for i,SS in enumerate(processing_SS):
                if atom1 in SS:
                    left_idx=i
            for j,SS in enumerate(processing_SS):
                if atom2 in SS:
                    right_idx=j
            #print(i,j)
            if left_idx!=right_idx and [left_idx,right_idx] not in edge and [right_idx,left_idx] not in edge:
                edge.append(sorted([left_idx,right_idx]))
                edge_atoms.update({str(sorted([left_idx,right_idx])):[atom1,atom2]})
        
        #further fix edges in substructure-based graph
        del_idx=[]
        del_edge=[]
        for atom1,atom2 in edge:
            if [atom1,atom2] in del_edge or [atom2,atom1] in del_edge:
                continue
            for atom3,atom4 in edge:
                if [atom3,atom4] in del_edge or [atom4,atom3] in del_edge:
                    continue
                if atom1==atom3 and [atom1,atom2]!=[atom3,atom4]:
                    if [atom2,atom4] in edge:
                        del_idx.append(edge.index([atom2,atom4]))
                        del_edge.append([atom2,atom4])
                    if [atom4,atom2] in edge:
                        del_idx.append(edge.index([atom4,atom2]))
                        del_edge.append([atom4,atom2])
                elif atom1==atom4 and [atom1,atom2]!=[atom3,atom4]:
                    if [atom2,atom3] in edge:
                        del_idx.append(edge.index([atom2,atom3]))
                        del_edge.append([atom2,atom3])
                    if [atom3,atom2] in edge:
                        del_idx.append(edge.index([atom3,atom2]))
                        del_edge.append([atom3,atom2])
                elif atom2==atom3 and [atom1,atom2]!=[atom3,atom4]:
                    if [atom1,atom4] in edge:
                        del_idx.append(edge.index([atom1,atom4]))
                        del_edge.append([atom1,atom4])
                    if [atom4,atom1] in edge:
                        del_idx.append(edge.index([atom4,atom1]))
                        del_edge.append([atom4,atom1])
                elif atom2==atom4 and [atom1,atom2]!=[atom3,atom4]:
                    if [atom1,atom3] in edge:
                        del_idx.append(edge.index([atom1,atom3]))
                        del_edge.append([atom1,atom3])
                    if [atom3,atom1] in edge:
                        del_idx.append(edge.index([atom3,atom1]))
                        del_edge.append([atom3,atom1])
        del_idx=list(set(del_idx))
        for i in sorted(del_idx,reverse=True):
            del edge_atoms[str(edge[i])]
        for i in del_edge:
            try:
                edge.remove(i)
            except:
                pass
        edge=np.array(edge)
        
        #if there is no edge in substructure-based graph(only one node)....
        if len(edge)==0:
            f2f=(torch.tensor([0]).long(),torch.tensor([0]).long())
        else:
            f2f=(torch.tensor(edge[:,0]).long(),torch.tensor(edge[:,1]).long())
        return a2f_edges,f2a_edges,num_funcs,f2f,edge_atoms
def SS_bond_feature_generation(f2a_edges,bond_feature,num_funcs,u,v):
    """generate bond features for substructure"""
    #output:bond features of substructures
    processing_SS=[]
    SS_bond_f_list=[]
    for i in range(num_funcs):
        processing_SS.append(list((f2a_edges[1][np.argwhere(f2a_edges[0].numpy()==i).ravel()]).numpy()))
    
    for SS in processing_SS:
        SS_bond_feature=torch.zeros(13)
        atom_index_list=[]
        for atom in SS:
            for nei_atom,index in zip(v[u==atom],np.argwhere(u==atom)):
                if nei_atom in SS and [atom,nei_atom] not in atom_index_list and [nei_atom,atom] not in atom_index_list:
                    atom_index_list.append([atom,nei_atom])
                    #print(bond_feature[index[0]].shape)
                    SS_bond_feature+=bond_feature[index[0]]
        SS_bond_f_list.append(SS_bond_feature)
    return torch.stack(SS_bond_f_list,dim=0)

        
def smiles_to_Molgraph(smiles,
                         add_self_loop=False,
                         node_featurizer=ATOM_FEATURIZER,
                         edge_featurizer=BOND_FEATURIZER,
                         canonical_atom_order=False,
                         explicit_hydrogens=False,
                         use_cycle: bool = True,
                         num_virtual_nodes=0):
    """Construct atom-based graph and substructure-based graph of a molecule"""
    mol = Chem.MolFromSmiles(smiles)
    g = mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer,
                       canonical_atom_order, explicit_hydrogens, num_virtual_nodes)
    
    if g is None:
        return None
    nf = torch.cat([g.ndata[nf_field] for nf_field in NODE_ATTRS], dim=-1)
    try:
        ef = torch.cat([g.edata[ef_field] for ef_field in EDGE_ATTRS], dim=-1)
    except KeyError:  # Ionic bond only case.
        return None
    nx_multi_g = g.to_networkx(node_attrs=NODE_ATTRS, edge_attrs=EDGE_ATTRS).to_undirected()
    nx_g = nx.Graph(nx_multi_g)
    incidence_info = get_substructures([nx_g],
                               use_cycle=use_cycle)[0]
    atg, gta = incidence_info.unbind(dim=0)
    num_func_groups = int(gta.max()) + 1
    a2f_edges = (atg.long(), gta.long())
    f2a_edges = (gta.long(), atg.long())
    u, v = g.edges()
    """construct edge between substructures"""
    a2f_edges,f2a_edges,num_func_groups,f2f,edge_atoms=SSgraph_edge_construction(smiles,f2a_edges,num_func_groups)
   
    
    mol_g = dgl.heterograph({
        ('atom', 'interacts', 'atom'): (u.long(), v.long()),
        ('atom', 'a2f', 'func_group'): a2f_edges,
        ('func_group', 'f2a', 'atom'): f2a_edges,
        ('func_group', 'interacts', 'func_group'): f2f,
        ('molecule','interacts','molecule'):(torch.tensor([0]),torch.tensor([0]))
    })
    
    f_atom_func_group=nf[f2a_edges[1][np.argwhere(f2a_edges[0].numpy()==0).ravel()],:].sum(0)
    f_atom_func_group=torch.tensor(np.vstack(f_atom_func_group).reshape(-1,1)).t()
    for i in range(1,num_func_groups):
        cat_feat=torch.tensor(np.vstack(nf[f2a_edges[1][np.argwhere(f2a_edges[0].numpy()==i).ravel()],:].sum(0)).reshape(-1,1)).t()
        f_atom_func_group=torch.cat((f_atom_func_group,cat_feat),dim=0)
        
    #generate bond features for substructures
    f_bond_func_group=SS_bond_feature_generation(f2a_edges,ef,num_func_groups,u.numpy(),v.numpy())
    
    #cat atom features and bond features to obtain the final substructure feature
    f_func_group=torch.cat((f_atom_func_group,f_bond_func_group),dim=1)
    
    #
    if len(mol_g.nodes('atom')) != nf.shape[0]:  # when a certain atom is not connected to the other atoms.
        return None
    mol_g.nodes['atom'].data['feat'] = nf
    
    mol_g.edges[('atom', 'interacts', 'atom')].data['feat'] = ef

    
    """construct edge feature for substructure-based graph"""
    edge_atom_features_list=[]
    for atoms in edge_atoms.values():
        edge_atom_features_list.append(nf[atoms].sum(0).ravel())
        
    if len(edge_atom_features_list)==0:
        edge_atom_features=torch.stack([torch.zeros(37)],dim=0)
    else:
        edge_atom_features=torch.stack(edge_atom_features_list,dim=0)
    f_func_edge=edge_atom_features
    mol_g.edges[('func_group', 'interacts', 'func_group')].data['feat'] = f_func_edge
    mol_g.nodes['func_group'].data['feat'] =f_func_group
    fingerprints = MACCSkeys.GenMACCSKeys(mol)
    #mol_feature=list(map(lambda x:int(x),list(fingerprints)))
    #mol_g.nodes['molecule'].data['feat']=torch.stack([torch.tensor(mol_feature,dtype=torch.float32)],dim=0).float()
    maccs = list(map(int, list(fingerprints)))
    alerts = pd.read_csv("alert_collection.csv")
    alerts = alerts[(alerts["rule_set_name"] == "LINT")]

    keys = list(alerts["description"])
    values = list(alerts["smarts"])

    global TOXIC_SMARTS

    TOXIC_SMARTS = dict(zip(keys, values))



    global_features = torch.tensor(maccs + get_global_features(smiles), dtype=torch.float32).unsqueeze(0)
    mol_g.nodes['molecule'].data['feat'] = global_features

    return mol_g
#smiles_to_Molgraph('Fc1cc(cc(F)c1)C[C@H](NC(=O)c1cc(cc(c1)C)C(=O)N(CCC)CCC)[C@H](O)[C@@H]1[NH2+]C[C@H](Oc2ccccc2)C1')
