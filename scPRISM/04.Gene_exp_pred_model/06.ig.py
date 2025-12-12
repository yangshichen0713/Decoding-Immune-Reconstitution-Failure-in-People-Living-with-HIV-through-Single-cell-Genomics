
import warnings
warnings.filterwarnings('ignore')
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import sparse
from module import *
def setup_seed(seed):

    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False  
    torch.set_float32_matmul_precision('high')
    print("seed set ok!")
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class scDataset(Dataset):
    def __init__(self,index,gene_idx):
        self.path='/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/adata_process'
        self.index_list=index
        self.gene_idx=gene_idx
    def __len__(self):
        return len(self.index_list)
    def get_np_array(self, filename):
        return np.load(os.path.join(self.path, filename))
    def __getitem__(self, idx):
        index_name=self.index_list[idx]
        array_idx=self.get_np_array(str(index_name)+'.npy')
        gene = torch.tensor(array_idx[:582], dtype=torch.bfloat16)
        peak = torch.tensor(array_idx[582:], dtype=torch.bfloat16)

        mask=torch.zeros(582)
        mask[self.gene_idx]=1
        return gene, peak, mask

import pickle
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")
    
# 定义一个函数，用于加载文件中的数据
# 定义一个函数，用于加载文件中的数据
    # 打开文件，以二进制模式读取
def load_data(filename):
        # 使用pickle模块加载文件中的数据
    with open(filename, 'rb') as f:
    # 返回加载的数据
        data = pickle.load(f)
    return data

import torch
import torch.nn as nn

class Peak2GeneModel_Gene(nn.Module):
    def __init__(self, input_dim=64,hidden_dim=512,out_features=582,choose_gene_idx=185):
        super().__init__()
        # 定义峰编码器
        self.peak_encoder= TokenizedFAEncoder(5583, 64, True, 7, 0.1, 'layernorm')
        # 定义投影层
        self.projection_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # 定义解码器
        self.decoder = GatedMLP(64, out_features=out_features)
        self.matrix=torch.load('./data/mask_mat.pt')
        self.choose_gene_idx=choose_gene_idx
    def forward(self,peak,mask):
        matrix=torch.tensor(self.matrix,device=mask.device)
        mask=torch.mm(mask,matrix)
        mask = torch.cat((torch.zeros(size=(mask.shape[0],1), dtype=torch.bfloat16,device=mask.device), mask), dim=1)
        
        # 计算掩码
        peak_embed = self.peak_encoder(peak,mask)
        peak_embed = self.projection_layer(peak_embed)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            peak_embed = (peak_embed * m).sum(1) / m.sum(1) 
        pred_gene = self.decoder(peak_embed)
        return pred_gene[:,self.choose_gene_idx]

setup_seed(3407)
dict_fen=load_data('./data/fen.pkl')

X_test = dict_fen['test']

# gene_choose=load_data('/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/ig/dict_gene_ig.pkl')

# content_list=gene_choose['content_list_index']

from captum.attr import IntegratedGradients

from tqdm import tqdm
list_gene_all=load_data('./data/list_gene_all.pkl')

##归因指定的基因
choose_gene_name=['NFKBIA', 'TNFAIP3', 'FOS', 'FOSB', "KLF6"]
choose_gene_idx=[list_gene_all.index(i) for i in choose_gene_name]


with torch.cuda.amp.autocast():
    for gene_idx in choose_gene_idx:
        test_dataset = scDataset(X_test,gene_idx=gene_idx)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)
        list_out=[]
        model=Peak2GeneModel_Gene(choose_gene_idx=gene_idx)
        model.load_state_dict(torch.load('./model/hiv_model-epoch=98-val_loss=0.4079.ckpt')['state_dict'])
        model=model.to('cuda:0')
        ig = IntegratedGradients(model)
        
        for i in tqdm(test_loader): 
            gene, peak, mask = i
            peak = peak.to('cuda:0')
            mask = mask.to('cuda:0')
            attributions = ig.attribute(peak, additional_forward_args=mask,n_steps=50,baselines=0.0)
            a=attributions
            a=a.float().cpu().detach().numpy()
            list_out.append(a)
            del peak,mask,attributions
        del model    
        save_data(list_out,'./ig/ig_output/'+str(gene_idx)+'.pkl')
