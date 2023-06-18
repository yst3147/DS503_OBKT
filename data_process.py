import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import match_seq_len

class eedi_a(Dataset):
    def __init__(self, seq_len, method, dataset_dir="data/eedi_a/") -> None:
        super().__init__()
        
        self.q_matrix = pd.read_pickle(dataset_dir + "q_matrix_eedi_a.pkl")
        self.dataset_dir = dataset_dir+method
 
        self.dataset_path = os.path.join(
        self.dataset_dir, "eedi_a.pkl"
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "rb") as f: 
                self.o_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.o_seqs, self.q_list, self.u_list = self.preprocess()

        
        self.num_q = max(max(sub_lst) for sub_lst in self.q_seqs)+1
        

        if seq_len:
            self.q_seqs, self.r_seqs, self.o_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.o_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.o_seqs[index]

    def __len__(self):
        return self.len
    
    def preprocess(self):
        df = pd.read_pickle(self.dataset_path).sort_values(by=["DateAnswered"])
        

        u_list = np.unique(df["UserId"].values)
        q_list = np.unique(df["QuestionId"].values)

        q_seqs = []
        r_seqs = []
        o_seqs = []

        grouped = df.groupby(df.iloc[:, 0])
        for student, group in grouped:
            q_seqs.append(group.iloc[:, 1].values)
            r_seqs.append(group.iloc[:, 6].values)
            o_seqs.append(group.iloc[:, 7].values)


        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "wb") as f:
            pickle.dump(o_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)

        return q_seqs, r_seqs, o_seqs, q_list, u_list

class eedi_b(Dataset):
    def __init__(self, seq_len, method, dataset_dir="data/eedi_b/") -> None:
        super().__init__()
        
        self.q_matrix = pd.read_pickle(dataset_dir + "q_matrix_eedi_b.pkl")
        self.dataset_dir = dataset_dir+method


        self.dataset_path = os.path.join(
        self.dataset_dir, "eedi_b.pkl"
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "rb") as f: 
                self.o_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)

        else:
            self.q_seqs, self.r_seqs, self.o_seqs, self.q_list, self.u_list = self.preprocess()

        self.num_q = max(max(sub_lst) for sub_lst in self.q_seqs)+1
        

        if seq_len:
            self.q_seqs, self.r_seqs, self.o_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.o_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.o_seqs[index]

    def __len__(self):
        return self.len
    
    def preprocess(self):
        df = pd.read_pickle(self.dataset_path).sort_values(by=["DateAnswered"])
        
        u_list = np.unique(df["UserId"].values)
        q_list = np.unique(df["QuestionId"].values)

        q_seqs = []
        r_seqs = []
        o_seqs = []

        grouped = df.groupby(df.iloc[:, 0])
        for student, group in grouped:
            q_seqs.append(group.iloc[:, 1].values)
            r_seqs.append(group.loc[:, 'IsCorrect'].values)
            o_seqs.append(group.loc[:, 'OptionWeight'].values)


        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "wb") as f:
            pickle.dump(o_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)

        return q_seqs, r_seqs, o_seqs, q_list, u_list   
    
class ednet(Dataset):
    def __init__(self, seq_len, method, dataset_dir="data/ednet/") -> None:
        super().__init__()
        
        self.q_matrix = pd.read_pickle(dataset_dir + "q_matrix_ednet.pkl")
        self.dataset_dir = dataset_dir+method

        self.dataset_path = os.path.join(
        self.dataset_dir, "ednet.pkl"
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "rb") as f: 
                self.o_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.o_seqs, self.q_list, self.u_list = self.preprocess()

        self.num_q = max(max(sub_lst) for sub_lst in self.q_seqs)+1
        

        if seq_len:
            self.q_seqs, self.r_seqs, self.o_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.o_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.o_seqs[index]

    def __len__(self):
        return self.len
    
    def preprocess(self):
        df = pd.read_pickle(self.dataset_path)

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["question_id"].values)

        q_seqs = []
        r_seqs = []
        o_seqs = []

        grouped = df.groupby(df.iloc[:, 0])
        for student, group in grouped:
            q_seqs.append(group.iloc[:, 1].values)
            r_seqs.append(group.loc[:, 'corretness'].values)
            o_seqs.append(group.loc[:, 'OptionWeight'].values)


        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "o_seqs.pkl"), "wb") as f:
            pickle.dump(o_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)

        return q_seqs, r_seqs, o_seqs, q_list, u_list

