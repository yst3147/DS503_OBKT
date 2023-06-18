import numpy as np
import torch
import os
from torch.nn import Module, Parameter, Embedding, Linear, Transformer,init
from torch.nn.init import normal_
from torch.nn.functional import one_hot, mse_loss, binary_cross_entropy,l1_loss
from sklearn import metrics

class SAINT(Module):
    def __init__(
        self, num_q, q_matrix,option, n, d, num_attn_heads, dropout, num_tr_layers=1
    ):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_tr_layers = num_tr_layers

        self.E = Embedding(self.num_q*2, self.d)
        self.R = Embedding(2, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        self.S = Parameter(torch.Tensor(1, self.d))

        self.Linear = Linear(self.d*3,self.d)
        self.option_emb = Embedding(5,self.d)
        self.q_matrix = q_matrix
        normal_(self.P)
        normal_(self.S)

        self.transformer = Transformer(
            self.d,
            self.num_attn_heads,
            num_encoder_layers=self.num_tr_layers,
            num_decoder_layers=self.num_tr_layers,
            dropout=self.dropout,
        )
        self.Linear1 = Linear(self.d*2,self.d)
        self.Linear2 = Linear(self.d*3,self.d)
        
        self.pred = Linear(self.d, self.num_q)
  
        self.KC = Parameter(torch.Tensor(self.q_matrix.shape[1], self.d))

        self.fc2 = Linear(self.num_q,1)
        self.linear = Linear(self.d, self.d)
        self.option = option
    def forward(self, q, r,o):
        batch_size,seq_len = r.shape

        E = self.E(q)

        # baseline : no option + no KC
        if self.option == 'no':
            E = E

        # option
        elif self.option == 'no_kc':   
            E = torch.cat((E, self.option_emb(o)), 2)
            E = self.Linear1(E)

        # kc
        elif self.option == 'no_option': 
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)
            E = torch.cat((E, kc_parameter), 2)
            E = self.Linear1(E)

        # option + kc
        else:
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)
            E = torch.cat((E, kc_parameter,self.option_emb(o)), 2)
            E = self.Linear2(E)
        
        E = E.reshape(batch_size,seq_len,-1).permute(1, 0, 2)


        R = self.R(r[:, :-1]).permute(1, 0, 2)
        S = self.S.repeat(batch_size, 1).unsqueeze(0)
        R = torch.cat([S, R], dim=0)

        P = self.P.unsqueeze(1).cuda()


        mask = self.transformer.generate_square_subsequent_mask(self.n).cuda()

        R = self.transformer(
            E + P, R + P, mask, mask, mask
        )
        R = R.permute(1, 0, 2)

        R = self.pred(R)


        return torch.sigmoid(R).squeeze()


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
                h = self(q.long(), r.long(),o.long())
                y = (h * one_hot(qshft.long(), self.num_q)).sum(-1)
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
                    h= self(q.long(), r.long(),o.long())
                    y = (h * one_hot(qshft.long(), self.num_q)).sum(-1)
                    y = torch.masked_select(y, m).detach()
                    t = torch.masked_select(r, m).detach()
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


