import os

import numpy as np
import torch
from tqdm import tqdm
from torch.nn import Module, Embedding, LSTM, GRU,Linear, Dropout, Sequential,ReLU,MultiheadAttention,Sigmoid,Parameter,init
from torch.nn.functional import one_hot, binary_cross_entropy,mse_loss,l1_loss
from sklearn import metrics
import matplotlib.patheffects as path_effects
from torch.nn.init import kaiming_normal_

import torch.nn as nn
import matplotlib.pyplot as plt

class DKT(Module):

    def __init__(self, num_q,q_matrix,option, emb_size, hidden_size):
        super().__init__()
        self.q_matrix = q_matrix
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.interaction_emb = Embedding(int(self.num_q*2), self.emb_size)
        self.option_emb = Embedding(5,self.emb_size)
        self.lstm_layer1 = LSTM(self.emb_size*1, self.hidden_size, batch_first=True)
        self.lstm_layer2 = LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
        self.lstm_layer3 = LSTM(self.emb_size*3, self.hidden_size, batch_first=True)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.sigmoid = Sigmoid()
        self.dropout_layer = Dropout()
        self.KC = nn.Parameter(torch.Tensor(self.q_matrix.shape[1], self.hidden_size))
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.num_q,1)
        self.option = option

    def forward(self, q, r, o):

        x = q + self.num_q * r

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

            h = torch.cat((self.interaction_emb(x),kc_parameter), 2)
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
        aucs,loss_means,loss_mean_test = [],[],[]
        max_auc = 0
        min_loss = 0
        for i in range(1, num_epochs + 1):
            loss_mean = 0
            loss_mean_test = 0
            for k,data in enumerate((train_loader)):
                q, r, o, qshft, rshft, oshft, m = data
                self.train()
                h= self(q.long(), r.long(),o.long())
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


        values = [0, 1]
        probabilities = [0.4, 0.6]  # 0 or 1 choose probability
        np.random.seed(1456)
        q_ = torch.tensor(np.random.choice(range(7), size=21))
        r_ = torch.tensor(np.random.choice(values, p=probabilities, size=21))

        o_A_ = []
        for num in r_:
            if num == 1:
                o_A_.append(4)
            else:
                o_A_.append(np.random.choice([1, 2, 3]))

        q = q_.unsqueeze(0)
        print(q.size())
        r = r_.unsqueeze(0)
        o_A = torch.tensor(o_A_).unsqueeze(0)

        with torch.no_grad():
            self.eval()
            y_A= self(q.long().cuda(), r.long().cuda(), o_A.long().cuda())
        data = y_A.squeeze().cpu().detach().numpy()[:,[1,2,3,4,5,6,7]].T
        fig, ax = plt.subplots(figsize=(10, 5))
        cbar_shrink = 0.2

        heatmap = ax.imshow(data, interpolation='nearest', cmap="YlGnBu")

        rows, cols = data.shape
        q2 = q.squeeze(0).detach().cpu().tolist() 
        for j in range(cols):
            for i in range(rows):
                text = ax.text(j, i, round(data[i, j],2),
                            ha='center', va='center', color='black', fontsize=6)
                    
                if q2[j] == i:
                    text.set_path_effects([path_effects.Stroke(linewidth=0.4, foreground='black'),
                            path_effects.Normal()])
        # add colorbar
        cbar = plt.colorbar(heatmap, ax=ax, shrink=cbar_shrink)
        # save heatmap
        plt.savefig('dkt'+"_"+self.option+'_heatmap.png')

        if self.option != 'no':
            np.random.seed(445)
            o_B_ = []
            for num in r_:
                if num == 1:
                    o_B_.append(4)
                else:
                    o_B_.append(np.random.choice([1, 2, 3]))

            o_B = torch.tensor(o_B_).unsqueeze(0)
            with torch.no_grad():
                self.eval()
                y_B= self(q.long().cuda(), r.long().cuda(), o_B.long().cuda())
            data = y_B.squeeze().cpu().detach().numpy()[:,[1,2,3,4,5,6,7]].T
            fig, ax = plt.subplots(figsize=(10, 5))
            cbar_shrink = 0.2

            heatmap = ax.imshow(data,interpolation='nearest', cmap="YlGnBu")
            q2 = q.squeeze(0).detach().cpu().tolist() 
            rows, cols = data.shape
            for j in range(cols):
                for i in range(rows):
                    text = ax.text(j, i, round(data[i, j],2),
                                ha='center', va='center', color='black', fontsize=6)
                    
                    if q2[j] == i:
                        text.set_path_effects([path_effects.Stroke(linewidth=0.4, foreground='black'),
                            path_effects.Normal()])
            # add colorbar
            cbar = plt.colorbar(heatmap, ax=ax, shrink=cbar_shrink)
            # save heatmap
            plt.savefig('dkt'+"_"+self.option+'_B_heatmap.png')
        return max_auc, min_loss