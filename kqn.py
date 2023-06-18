import os
from sklearn import metrics
from tqdm import tqdm
import numpy as np

import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU
from torch.nn.functional import binary_cross_entropy,l1_loss
import torch.nn as nn

class KQN(Module):
    def __init__(self, num_q,q_matrix,option, dim_s):
        super().__init__()
        self.num_q = num_q
        self.dim_v = dim_s
        self.dim_s = dim_s
        self.hidden_size = dim_s

        self.x_emb = Embedding(self.num_q * 2, self.dim_v)
        self.knowledge_encoder = LSTM(self.dim_v, self.dim_v, batch_first=True)
        self.out_layer = Linear(self.dim_v, self.dim_s)
        self.dropout_layer = Dropout()

        self.q_emb = Embedding(self.num_q, self.dim_v)
        self.skill_encoder = Sequential(
            Linear(self.dim_v, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.dim_v),
            ReLU()
        )
        self.skill_encoder2 = Sequential(
            Linear(self.dim_v, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.num_q),
            ReLU()
        )
        self.q_matrix = q_matrix

        self.KC = nn.Parameter(torch.Tensor(self.q_matrix.shape[1], self.dim_s))
        self.fc = nn.Linear(self.dim_s, self.num_q)
        self.fc2 = nn.Linear(self.num_q,1)
        self.linear = nn.Linear(self.dim_s, self.dim_s)
        self.linear1 = nn.Linear(self.dim_s*2, self.dim_s)
        self.linear2 = nn.Linear(self.dim_s*3, self.dim_s)
        self.option = option
        self.option_emb = nn.Embedding(5,self.dim_s)

    def forward(self, q, r,o, qry):
        # Knowledge State Encoding
        x = q + self.num_q * r
        x = self.x_emb(x)

        # baseline : no option + no KC
        if self.option == 'no':
            h, _ = self.knowledge_encoder(x)

        # option
        elif self.option == 'no_kc':   
            x= self.linear1(torch.cat((x, self.option_emb(o)), 2))
            h, _ = self.knowledge_encoder(x)

        # kc
        elif self.option == 'no_option': 
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)

            x= self.linear1(torch.cat((x, kc_parameter), 2))
            h, _ = self.knowledge_encoder(x)
        
        # option + kc
        else:   
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)

            x= self.linear2(torch.cat((x, kc_parameter, self.option_emb(o)), 2))
            h, _ = self.knowledge_encoder(x)


        ks = self.out_layer(h)
        ks = self.dropout_layer(ks)

        e = self.q_emb(qry)
        o = self.skill_encoder(e)
        s = o / torch.norm(o, p=2)

        p = torch.sigmoid((ks * s).sum(-1))



        return p

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        aucs,loss_means,loss_mean_test = [],[],[]
        max_auc = 0
        min_loss = 0
        for i in range(1, num_epochs + 1):
            loss_mean = 0
            loss_mean_test=0
            for k,data in enumerate((train_loader)):
                q, r, o, qshft, rshft, oshft, m = data

                self.train()
                y = self(q.long(), r.long(),o.long(),qshft.long())
                #y = (h * one_hot(qshft.long(), self.num_q)).sum(-1)
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
                    y = self(q.long(), r.long(),o.long(),qshft.long())
                    #y = (h * one_hot(qshft.long(), self.num_q)).sum(-1)
                    y = torch.masked_select(y, m).detach()
                    t = torch.masked_select(rshft, m).detach()
                    loss = l1_loss(y, t)
                    loss_mean_test += loss.data
                    y_true += t.cpu().tolist() 
                    y_score += y.cpu().tolist()
                loss_mean_test = loss_mean_test/kk
            auc = metrics.roc_auc_score(y_true,y_score)
            y_score =np.round(y_score)
        
            if auc > max_auc:
                torch.save(self.state_dict(),os.path.join(ckpt_path, "model.ckpt"))
                max_auc = auc
                min_loss = loss_mean_test
            print("Epoch: {}, Train Loss : {:.4f}, Test Loss: {:.4f}, AUC: {:.4f}".format(i,loss_mean, loss_mean_test, auc ))
        return max_auc, min_loss
