import os
import numpy as np
import torch
from tqdm import tqdm

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout,init
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy,one_hot,mse_loss,l1_loss
from sklearn import metrics
from sklearn.metrics import f1_score,recall_score
import random
import numpy as np
import torch


class SAKT(Module):

    def __init__(self, num_q, q_matrix,option,n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.drop = Dropout(self.dropout)
        self.FFN_layer_norm = LayerNorm(self.d)
        self.out_layer = Linear(self.d, self.num_q)
        self.option_emb = Embedding(5,self.d)
        self.q_matrix = q_matrix
        self.Linear2 = Linear(self.d*3,self.d)

        self.Linear1 = Linear(self.d*2,self.d)
        self.KC = Parameter(torch.Tensor(self.q_matrix.shape[1], self.d))
        self.fc = Linear(self.d*2, self.num_q)
        self.fc2 = Linear(self.num_q,1)
        self.linear = Linear(self.d, self.d)
        self.option = option
    def forward(self, q, r, o, qry):

        num_skill = self.q_matrix.shape[1]

        batch_size,seq_len = q.size()
        x = q + self.num_q * r
        

        M = self.M(x)

        # baseline : no option + no KC
        if self.option == 'no':
            M = M

        # option
        elif self.option == 'no_kc':   
            M = torch.cat((M, self.option_emb(o)), 2)
            M = self.Linear1(M)

        # kc
        elif self.option == 'no_option': 
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)
            M = torch.cat((M, kc_parameter), 2)
            M = self.Linear1(M)

        # option + kc
        else:
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)
            M = torch.cat((M, kc_parameter,self.option_emb(o)), 2)
            M = self.Linear2(M)

        M = M.reshape(batch_size,seq_len,-1).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(torch.ones([E.shape[0], M.shape[0]]), diagonal=1).bool()

        M = M + P
        
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)

        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)      
        
        S = self.attn_layer_norm(S + M + E)


        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.out_layer(F)).squeeze()

        return p

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        aucs,loss_means = [],[]
        max_auc = 0
        min_loss = 0
        for i in range(1, num_epochs + 1):
            loss_mean = 0
            loss_mean_test = 0
            for k,data in enumerate((train_loader)):
                q, r, o, qshft, rshft, oshft, m = data

                self.train()
                heatmap = self(q.long(), r.long(),o.long(),qshft.long())
                y = (heatmap * one_hot(qshft.long(), self.num_q)).sum(-1)
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = l1_loss(y, t)

                loss.backward()
                opt.step()
                loss_mean += loss.data
            loss_mean = loss_mean/k

            with torch.no_grad():
                y_true, y_score = [],[]
                for kk,data in enumerate(test_loader):
                    q, r, o, qshft, rshft, oshft, m = data

                    self.eval()
                    heatmap= self(q.long(), r.long(),o.long(),qshft.long())
                    y = (heatmap * one_hot(qshft.long(), self.num_q)).sum(-1)
                    y = torch.masked_select(y, m).detach()
                    t = torch.masked_select(rshft, m).detach()
                    y_true += t.cpu().tolist() 
                    y_score += y.cpu().tolist()
                    loss = l1_loss(y, t)
                    loss_mean_test += loss.data
                loss_mean_test = loss_mean_test/kk
            auc = metrics.roc_auc_score(y_true,y_score)
            y_score =np.round(y_score)
        
            if auc > max_auc:
                torch.save(self.state_dict(),os.path.join(ckpt_path, "model.ckpt"))
                max_auc = auc
                min_loss = loss_mean_test
            print("Epoch: {}, Train Loss : {:.4f}, Test Loss: {:.4f}, AUC: {:.4f}".format(i,loss_mean, loss_mean_test, auc ))
        return max_auc, min_loss





