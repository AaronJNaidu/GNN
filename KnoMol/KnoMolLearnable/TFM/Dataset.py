import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch


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
        data_list = []
        data_len = len(xd)
        print('number of data ', data_len)
        for i in range(data_len):
            smiles = xd[i]
            if smiles is not None:
                labels = np.asarray([y[i]])
                leng, features, edge_index, edge_attr, ringm, aromm, alipm, hetem, adj_order_matrix, dis_order_matrix = smile_graph[smiles]
                if len(edge_index) > 0:
                    GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0).contiguous(), edge_attr=torch.Tensor(edge_attr), y=torch.FloatTensor(labels))
                    GCNData.leng = [leng] 
                    GCNData.adj = adj_order_matrix
                    GCNData.dis = dis_order_matrix
                    GCNData.ringm = ringm
                    GCNData.aromm = aromm
                    GCNData.alipm = alipm
                    GCNData.hetem = hetem
                    GCNData.smi = smiles

                    adj_map = adj_order_matrix  # 2D list of strings
                    L = leng

                    label2int = {
                        None: 0,
                        'bond_single': 1,
                        'bond_single_conj': 2,
                        'bond_double': 3,
                        'bond_triple': 4,
                        'bond_aromatic': 5,
                        'bond_conjugated': 6,
                        'bond_distant' : 7
                    }

                    adj_label_int = np.zeros((L, L), dtype=np.int16)
                    localmask = np.zeros((L, L), dtype=np.uint8)
                    cojmask = np.zeros((L, L), dtype=np.uint8)

                    for r in range(L):
                        for c in range(L):
                            lab = adj_order_matrix[r][c]  # original string labels
                            if lab in label2int:
                                adj_label_int[r, c] = label2int[lab]
                            else:
                                adj_label_int[r, c] = 0
                            if lab in ('bond_single', 'bond_double', 'bond_triple', 'bond_aromatic'):
                                localmask[r, c] = 1
                            if lab in ('bond_single_conj', 'bond_double', 'bond_triple', 'bond_aromatic', 'bond_conjugated'):
                                cojmask[r, c] = 1


                    GCNData.adj_label = adj_label_int        # numpy array
                    GCNData.local_mask = localmask           # numpy array
                    GCNData.coj_mask = cojmask               # numpy array

                    data_list.append(GCNData)

                

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])