import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
from rdkit import Chem

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
                }'''



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



class MolNet(InMemoryDataset):
    def __init__(self, root='dataset', dataset=None, xd=None, y=None, transform=None, pre_transform=None, smile_graph=None):
        # root is required for save raw data and preprocessed data
        super(MolNet, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #pass
        return ['raw_file']

    @property
    def processed_file_names(self):
        return [self.dataset + '_pyg.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph):
        assert (len(xd) == len(y)), "smiles and labels must be the same length!"
        alerts = pd.read_csv("alert_collection.csv")
        alerts = alerts[(alerts["rule_set_name"] == "LINT")]

        keys = list(alerts["description"])
        values = list(alerts["smarts"])
       
        global TOXIC_SMARTS

        TOXIC_SMARTS = dict(zip(keys, values))
        data_list = []
        data_len = len(xd)
        print('number of data ', data_len)
        for i in range(data_len):
            smiles = xd[i]
            if smiles is not None:
                labels = np.asarray([y[i]])
                leng, features, edge_index, edge_attr, ringm, aromm, alipm, hetem, adj_order_matrix, dis_order_matrix = smile_graph[smiles]
                if len(edge_index) > 0:
                    global_feats = get_global_features(smiles)
                    if global_feats is None:
                        continue

                    #mol_features = torch.tensor(global_feats, dtype = torch.float)

                    GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0).contiguous(), edge_attr=torch.Tensor(edge_attr), y=torch.FloatTensor(labels))
                    GCNData.leng = [leng] 
                    GCNData.adj = adj_order_matrix
                    GCNData.dis = dis_order_matrix
                    GCNData.ringm = ringm
                    GCNData.aromm = aromm
                    GCNData.alipm = alipm
                    GCNData.hetem = hetem
                    GCNData.smi = smiles
                    GCNData.mol_features = torch.tensor(global_feats, dtype=torch.float).unsqueeze(0)
                   # print("mol_features shape:", GCNData.mol_features.shape)
                    data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
