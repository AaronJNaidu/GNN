import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
from torch_geometric.utils import to_dense_batch
from math import sqrt
from TFM.utils import get_attn_pad_mask, create_ffn
from torch_geometric.data import Batch

class Embed(nn.Module):
    def __init__(self, attn_head=4, output_dim=128, d_k=64, attn_layers=4, dropout=0.1, useedge=False, device='cuda:0', edge_attr_dim=6):
        super(Embed, self).__init__()
        self.device = device 

        self.relu = nn.ReLU()         
        self.edge = useedge
        self.edge_dim = edge_attr_dim
        self.layer_num = attn_layers 
        self.gnns = nn.ModuleList([GraphConv(37, output_dim) if i == 0 else GraphConv(output_dim, output_dim) for i in range(attn_layers)])
        self.nms = nn.ModuleList([nn.LayerNorm(output_dim) for _ in range(attn_layers)])
        self.dps = nn.ModuleList([nn.Dropout(dropout) for _ in range(attn_layers)])
        self.tfs = nn.ModuleList([Encoder(output_dim, d_k, d_k, 1, attn_head, dropout, edge_dim = self.edge_dim) for _ in range(attn_layers)])

        self.edge_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.edge_dim + 2 * output_dim, self.edge_dim),
                nn.ReLU(),
                nn.Linear(self.edge_dim, self.edge_dim)
            ) for i in range(attn_layers)
        ])

    def forward(self, x, edge_index, edge_attr, batch, leng, adj, dis, ring_masm, arom_masm, alip_masm, hete_mask, data=None):
        if self.edge:
            x = self.gnns[0](x, edge_index, edge_weight=edge_attr.norm(dim=1))
        else:
            x = self.gnns[0](x, edge_index)
        x = self.dps[0](self.nms[0](x))
        x = self.relu(x)

        x_batch, mask = to_dense_batch(x, batch)

        batch_size, max_len, output_dim = x_batch.size()
        matrix_pad = torch.zeros((batch_size, max_len, max_len))
        manu_mask_pad = torch.ones((batch_size, 6, max_len, max_len))
        edge_feat_pad = torch.zeros((batch_size, max_len, max_len, self.edge_dim), device=self.device)

        edge_idx = 0
        

        row, col = edge_index  
        graph_id = batch[row]  

        local_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        local_ptr = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i in range(batch.max().item() + 1):
            node_mask = (batch == i)
            local_index[node_mask] = torch.arange(node_mask.sum(), device=x.device)
            local_ptr[node_mask] = i

        u = local_index[row]  # local row indices per graph
        v = local_index[col]
        graph_id = local_ptr[row]  # batch index of each edge

        assert u.max().item() < edge_feat_pad.size(1), f"u.max={u.max().item()}, pad size={edge_feat_pad.size(1)}"
        assert v.max().item() < edge_feat_pad.size(2)
        assert graph_id.max().item() < edge_feat_pad.size(0)

        # Assign edge features
        edge_feat_pad[graph_id, u, v] = edge_attr
        edge_feat_pad[graph_id, v, u] = edge_attr  # symmetric


        for i, l in enumerate(leng):
            adj_ = torch.FloatTensor(adj[i])
            localmask = torch.where((adj_ >= 0.8) & (adj_ != 0.825), torch.ones_like(adj_), torch.zeros_like(adj_))
            cojmask = torch.where(adj_ > 0.8, torch.ones_like(adj_), torch.zeros_like(adj_))
            hete = torch.BoolTensor(hete_mask[i])
            hetemask = hete.unsqueeze(0) * hete.unsqueeze(-1)
            ring = torch.IntTensor(ring_masm[i])
            ring1 = ring.unsqueeze(0).repeat(ring.size(0), 1)
            ring2 = ring.unsqueeze(-1).repeat(1, ring.size(0))
            ringmask = torch.where((ring1 == ring2) & (ring1 > 0), torch.ones((ring.size(0), ring.size(0))), torch.zeros((ring.size(0),ring.size(0))))      
            arom = torch.IntTensor(arom_masm[i])
            arom1 = arom.unsqueeze(0).repeat(arom.size(0), 1)
            arom2 = arom.unsqueeze(-1).repeat(1, arom.size(0))
            arommask = torch.where((arom1 == arom2) & (arom1 > 0), torch.ones((arom.size(0), arom.size(0))), torch.zeros((arom.size(0),arom.size(0))))
            alip = torch.IntTensor(alip_masm[i])          
            alip1 = alip.unsqueeze(0).repeat(alip.size(0), 1)
            alip2 = alip.unsqueeze(-1).repeat(1, alip.size(0))
            alipmask = torch.where((alip1 == alip2) & (alip1 > 0), torch.ones((alip.size(0), alip.size(0))), torch.zeros((alip.size(0),alip.size(0))))       
            dis_ = torch.FloatTensor(dis[i])
            dis_ = torch.where(dis_ == 0, dis_, 1/(dis_))
            matrix = torch.where(adj_==0, dis_, adj_)
            matrix_pad[i, :int(l[0]), :int(l[0])] = matrix
            manu_mask_pad[i, :, :int(l[0]), :int(l[0])] = torch.cat([localmask.eq(0).unsqueeze(0),cojmask.eq(0).unsqueeze(0),hetemask.eq(0).unsqueeze(0),ringmask.eq(0).unsqueeze(0),arommask.eq(0).unsqueeze(0),alipmask.eq(0).unsqueeze(0)], 0)
        
        matrix_pad = matrix_pad.to(self.device)
        manu_mask_pad = manu_mask_pad.to(self.device)
        
        x_batch, matrix = self.tfs[0](x_batch, mask, matrix_pad, manu_mask_pad, edge_feat = edge_feat_pad)  
        for i in range(1, self.layer_num):
            x = torch.masked_select(x_batch, mask.unsqueeze(-1))
            x = x.reshape(-1, output_dim)

            if self.edge:
                # Update edge features: e_ij = MLP([e_ij, h_i, h_j])
                row, col = edge_index
                edge_input = torch.cat([edge_attr, x[row], x[col]], dim=1)
                edge_attr = self.edge_update[i](edge_input)
                x = self.gnns[i](x, edge_index, edge_weight=edge_attr.norm(dim=1))

            else:
                x = self.gnns[i](x, edge_index)
            x = self.dps[i](self.nms[i](x))

            x = self.relu(x)
            x_batch, mask = to_dense_batch(x, batch)
            x_batch, matrix = self.tfs[i](x_batch, mask, matrix, manu_mask_pad, edge_feat = edge_feat_pad)

        return x_batch


class Kno(nn.Module):
    def __init__(self, task='reg', tasks=1, attn_head=4, output_dim=128, d_k=64, attn_layers=4, D=16, dropout=0.1, useedge=False, device='cuda:0'):
        super(Kno, self).__init__()                                                                                                                    
        self.device = device
        self.emb = Embed(attn_head, output_dim, d_k, attn_layers, dropout, useedge, device)
        self.task = task
        # prediction module
        if task == 'clas':
            self.w1 = torch.nn.Parameter(torch.FloatTensor(D, output_dim))
            self.w2 = torch.nn.Parameter(torch.FloatTensor(2, D))
            self.th = nn.Tanh()
            self.sm = nn.Softmax(-1)
            self.bm = nn.BatchNorm1d(2, output_dim)
        elif task == 'reg':
            self.w1 = nn.Linear(output_dim, 1)
            self.sm = nn.Softmax(1)
            self.bm = nn.BatchNorm1d(output_dim)
        else:
            raise NameError('task must be reg or clas!')
        self.act = create_ffn(task, tasks, output_dim, dropout)
        self.reset_params()

    def reset_params(self):
        for weight in self.parameters():
            if len(weight.size()) > 1:
                nn.init.xavier_normal_(weight)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device)                                               # tensor
        leng, adj, dis = data.leng, data.adj, data.dis      # list
        ring_masm, arom_masm, alip_masm, hete_mask = data.ringm, data.aromm, data.alipm, data.hetem
        batch = data.batch.to(self.device)
        
        x_batch = self.emb(x, edge_index, edge_attr, batch, leng, adj, dis, ring_masm, arom_masm, alip_masm, hete_mask, data)

        if self.task == 'clas':
            x_bat = self.th(torch.matmul(self.w1, x_batch.permute(0, 2, 1)))  #  B O L
            x_bat = self.sm(torch.matmul(self.w2, x_bat))                     #  B X L
            x_p = torch.matmul(x_bat, x_batch)                                #  B X D
            x_p = self.bm(x_p)
            x_p = x_p.reshape(x_p.size(0), x_p.size(1)*x_p.size(2))
        else:
            x_p = self.bm(torch.sum(self.sm(self.w1(x_batch)) * x_batch, 1))

        # prediction
        logits = self.act(x_p)

        return logits


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout, edge_dim=None):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dp = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)
        self.edge_bias = nn.Sequential(nn.Linear(edge_dim, d_k), nn.Tanh(), nn.Linear(d_k, 1)) if edge_dim else None
        
    def forward(self, Q, K, V, attn_mask, edge_feat = None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(self.d_k)  

        if self.edge_bias is not None and edge_feat is not None:
            bias = self.edge_bias(edge_feat)
            bias = bias.squeeze(-1).unsqueeze(1)
            scores = scores + bias
        scores.masked_fill_(attn_mask, -1e9)         # Fills elements of self tensor with value where mask is True.
        attn = self.sm(scores)
        context = torch.matmul(self.dp(attn), V) # [batch_size, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout, edge_dim = None):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.W_V2 = nn.Linear(d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v*(n_heads+1), d_model, bias=False)
        self.nm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dp = nn.Dropout(p=dropout)
        self.sdpa = ScaledDotProductAttention(d_k, dropout, edge_dim=edge_dim)
        self.fu = nn.Sequential(
                    nn.LayerNorm(n_heads+1),
                    nn.Linear(n_heads+1, 6),
                    nn.ReLU(),
                    nn.Linear(6, 1),
                    nn.Sigmoid())
        
    def forward(self, input_Q, input_K, input_V, attn_mask, matrix, edge_feat=None):
        batch_size = input_Q.size(0)
        
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, max_len, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, max_len, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, max_len, d_v]

        context, attn = self.sdpa(Q, K, V, attn_mask, edge_feat = edge_feat)
        context2 = torch.matmul(matrix, self.W_V2(input_V)) 

        matrix = matrix * self.fu(torch.cat([matrix.unsqueeze(1), attn], 1).transpose(1,3)).squeeze()

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        context = torch.cat([context, context2], -1)
        output = self.fc(context)                                                          # [batch_size, max_len, d_model]
        
        return self.dp(self.nm(output)), matrix


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout, edge_dim):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout, edge_dim = edge_dim)
        self.nm = nn.LayerNorm(d_model)
        self.pos_ffn = nn.Sigmoid()

    def forward(self, enc_inputs, attn_mask, matrix, edge_feat):
        residual = enc_inputs
        enc_outputs, matrix = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask, matrix, edge_feat = edge_feat)
        enc_outputs = self.pos_ffn(enc_outputs)
        return self.nm(enc_outputs+residual), matrix


class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_layers, n_heads, dropout, edge_dim):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, dropout, edge_dim = edge_dim) for _ in range(n_layers)])
        self.nhead = n_heads

    def forward(self, enc_inputs, mask, matrix, manu_mask_pad, edge_feat):
        attn_mask = get_attn_pad_mask(mask)         
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead-manu_mask_pad.size(1), 1, 1)                   # attn_mask : [batch_size, n_heads, max_len, max_len]
        attn_mask = torch.cat([attn_mask, manu_mask_pad], 1).bool()
        for i, layer in enumerate(self.layers):
            enc_inputs, matrix = layer(enc_inputs, attn_mask, matrix, edge_feat)
        return enc_inputs, matrix
