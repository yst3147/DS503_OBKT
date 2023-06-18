import os
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, GRU,Linear, Dropout, Sequential,ReLU,MultiheadAttention,Sigmoid,Parameter,init
from torch.nn.functional import one_hot, binary_cross_entropy,mse_loss,l1_loss

from torch.nn.init import kaiming_normal_

# coding: utf-8

device = "cuda:0"

class ATKT(nn.Module):
    def __init__(self, num_q, q_matrix, option, skill_dim, answer_dim, hidden_dim):
        super(ATKT, self).__init__()
        self.num_q = num_q
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.rnn_1 = nn.LSTM(self.skill_dim*2, self.hidden_dim, batch_first=True)
        self.rnn_2 = nn.LSTM(self.skill_dim*3, self.hidden_dim, batch_first=True)
        self.rnn_3 = nn.LSTM(self.skill_dim*4, self.hidden_dim, batch_first=True)
        
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(int(self.num_q+1), self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0

        self.option_emb = nn.Embedding(5,self.answer_dim)
        self.option_emb.weight.data[0]= 0
        
        self.attention_dim = 30
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        self.q_matrix = q_matrix
        self.KC = nn.Parameter(torch.Tensor(self.q_matrix.shape[1], self.answer_dim))
        self.fc = nn.Linear(self.hidden_dim*2, self.num_q)
        self.fc2 = nn.Linear(self.num_q,1)
        self.linear = nn.Linear(self.answer_dim, self.answer_dim)
        self.option = option
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output


    def forward(self, skill, answer, option):
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        option_embedding=self.option_emb(option)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
        # baseline : no option + no KC
        if self.option == 'no':
            out, _ = self.rnn_1(skill_answer_embedding)
        
        # option
        elif self.option == 'no_kc':   
            out,_ = self.rnn_2(torch.cat((skill_answer_embedding, option_embedding), 2))
        
        # kc
        elif self.option == 'no_option': 
            q_e = self.q_matrix[skill].view(skill.shape[0], skill.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(skill.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)

            out,_ = self.rnn_2(torch.cat((skill_answer_embedding, kc_parameter), 2))
        
        # option + kc    
        else:
            q_e = self.q_matrix[skill].view(skill.shape[0], skill.shape[1],-1).float()
            kc_parameter1 = q_e.bmm(self.KC.repeat(skill.shape[0], 1, 1).float()).squeeze(1)
            kc_parameter2 = kc_parameter1/(q_e.sum(-1).unsqueeze(-1))
            kc_parameter = self.linear(kc_parameter2)

            out,_ = self.rnn_3(torch.cat((skill_answer_embedding, kc_parameter, option_embedding), 2))

        out=self.attention_module(out)
        out =self.fc(out)
        res = self.sig(out)

        return res.squeeze(), self.sig(self.fc2(out)).squeeze()




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
                h,y = self(q.long(), r.long(),o.long())
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
                    h,y = self(q.long(), r.long(),o.long())
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
        return max_auc, min_loss
