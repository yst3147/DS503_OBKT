import os
import numpy as np
from sklearn import metrics
from tqdm import tqdm

import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy,l1_loss
import torch.nn as nn


class DKTPlus(Module):
    def __init__(
        self, num_q, q_matrix, option, emb_size, hidden_size, lambda_r, lambda_w1, lambda_w2
    ):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer1 = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.lstm_layer2 = LSTM(self.emb_size * 2, self.hidden_size, batch_first=True)
        self.lstm_layer3 = LSTM(self.emb_size * 3, self.hidden_size, batch_first=True)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()
        self.q_matrix = q_matrix
        self.sigmoid = nn.Sigmoid()
        self.KC = nn.Parameter(torch.Tensor(self.q_matrix.shape[1], self.emb_size))
        self.linear = nn.Linear(self.emb_size, self.emb_size)
        self.linear2 = nn.Linear(self.emb_size * 3, self.emb_size)
        self.option = option
        self.option_emb = nn.Embedding(5, self.emb_size)

    def forward(self, q, r, o):
        batch_size, seq_len = q.size()
        x = q + self.num_q * r
        q_e = self.q_matrix[q].view(q.shape[0], q.shape[1], -1).float()
        kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
        kc_parameter2 = kc_parameter1 / (q_e.sum(-1).unsqueeze(-1))
        kc_parameter = self.linear(kc_parameter2)

        # baseline : no option + no KC
        if self.option == 'no':
            h, _ =  self.lstm_layer1(self.interaction_emb(x))
        
        # option
        elif self.option == 'no_kc':   
            h = torch.cat((self.interaction_emb(x), self.option_emb(o)), 2)
            h, _ =  self.lstm_layer2(h)

        # kc
        elif self.option == 'no_option':   
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)

            h = torch.cat((self.interaction_emb(x), kc_parameter), 2)
            h, _ =  self.lstm_layer2(h)
        
        # option + kc
        else:   
            q_e = self.q_matrix[q].view(q.shape[0], q.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(q.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)

            h = torch.cat((self.interaction_emb(x), kc_parameter, self.option_emb(o)), 2)
            h, _ =  self.lstm_layer3(h)

        y = self.dropout_layer(self.out_layer(h))
        y = self.sigmoid(y).squeeze()

        return y

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        aucs, loss_means, loss_mean_test = [], [], []
        max_auc = 0
        min_loss = 0
        for i in range(1, num_epochs + 1):
            loss_mean = 0
            loss_mean_test = 0
            for k, data in enumerate((train_loader)):
                q, r, o, qshft, rshft, oshft, m = data

                self.train()
                y = self(q.long(), r.long(), o.long())
                y_curr = (y * one_hot(q.long(), self.num_q)).sum(-1)
                y_next = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y_curr = torch.masked_select(y_curr, m)
                y_next = torch.masked_select(y_next, m)
                r = torch.masked_select(r, m)
                rshft = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = nn.L1Loss()(
                    y_next, rshft
                ) + self.lambda_r * nn.L1Loss()(y_curr, r)
                loss.backward()
                opt.step()

                loss_mean += loss.data
            loss_mean = loss_mean / k

            with torch.no_grad():
                y_true, y_score = [], []
                for kk, data in enumerate(test_loader):
                    q, r, o, qshft, rshft, oshft, m = data

                    self.eval()
                    y = self(q.long(), r.long(), o.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)
                    y = torch.masked_select(y, m).detach()
                    t = torch.masked_select(rshft, m).detach()

                    loss = nn.L1Loss()(y, t)
                    loss_mean_test += loss.data
                    y_true += t.cpu().tolist()
                    y_score += y.cpu().tolist()

                loss_mean_test = loss_mean_test / kk

            auc = metrics.roc_auc_score(y_true,y_score)
            y_score =np.round(y_score)
        
            if auc > max_auc:
                torch.save(self.state_dict(),os.path.join(ckpt_path, "model.ckpt"))
                max_auc = auc
                min_loss = loss_mean_test
            print("Epoch: {}, Train Loss : {:.4f}, Test Loss: {:.4f}, AUC: {:.4f}".format(i,loss_mean, loss_mean_test, auc ))
        return max_auc, min_loss