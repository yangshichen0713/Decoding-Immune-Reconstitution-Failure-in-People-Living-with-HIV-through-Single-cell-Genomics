import sys
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
setup_seed(3407)

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

from torch.utils.data import Dataset
class scDataset(Dataset):
    def __init__(self,index,mode):
        # 初始化函数，传入index和mode
        self.path='/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/adata_process'
        self.stage=np.load('/home/share/huadjyin/home/zhouxuanchi/HIV/new_atac_and_gene_to_class/HDs_INRs/data/stage.npy')
        self.index_list=index
        self.mode=mode
    def __len__(self):
        return len(self.index_list)
    def get_np_array(self, filename):
        return np.load(os.path.join(self.path, filename))
    def __getitem__(self, idx):
        index_name=self.index_list[idx]
        array_idx=self.get_np_array(str(index_name)+'.npy')
        
        tensor_all=torch.tensor(array_idx, dtype=torch.bfloat16)
        mask=torch.tensor((tensor_all[:582] != 0), dtype=torch.bfloat16)
        gene = tensor_all[:582]
        peak = tensor_all[582:]
        label=self.stage[index_name]
        if self.mode=='hds_or_inrs':
            if label=='HDs':
                label=torch.tensor(0,dtype=torch.float32)
            else:
                label=torch.tensor(1,dtype=torch.float32)
        elif self.mode=='hds_or_irs':
            if label=='HDs':
                label=torch.tensor(0,dtype=torch.float32)
            else:
                label=torch.tensor(1,dtype=torch.float32)
        elif self.mode=='irs_or_inrs':
            if label=='IRs':
                label=torch.tensor(0,dtype=torch.float32)
            else:
                label=torch.tensor(1,dtype=torch.float32)
        else:
            return  ValueError("some error.")
        return gene, peak,mask,label

dict_sample_stage=load_data('./data/dict_sample_stage.pkl')

test_list=['HD-H162','HD-H323','HD-H330','HD-H150','HD-H325','PD-H292','PD-H262','PD-H296','PD-H279','PD-H297','PD-H263','PD-H232','PD-H230','PD-H237','PD-H233']
hds_test=test_list[0:5]
inrs_test=test_list[10:15]
irs_test=test_list[5:10]
mode='hds_or_inrs'



class HIVModel(pl.LightningModule):
    def __init__(self, input_dim=64, hidden_dim=512):
        super().__init__()

        self.gene_encoder = TokenizedFAEncoder(582, 64, True, 7, 0.1, 'layernorm')
        self.peak_encoder = TokenizedFAEncoder(5583, 64, True, 7, 0.1, 'layernorm')
        self.decoder = GatedMLP(in_features=2*64, out_features=1)
        self.matrix=torch.load('/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/data/mask_mat.pt')
    def forward(self, gene, peak,mask_gene):   
        matrix=torch.tensor(self.matrix,device=mask_gene.device)
        mask_peak=torch.mm(mask_gene,matrix)
        mask_gene = torch.cat((torch.zeros(size=(mask_gene.shape[0],1), dtype=torch.bfloat16,device=mask_gene.device), mask_gene), dim=1)
        mask_peak = torch.cat((torch.zeros(size=(mask_peak.shape[0],1), dtype=torch.bfloat16,device=mask_gene.device), mask_peak), dim=1)
        #[B, 582] -> [B, 583, 64]
        gene = self.gene_encoder(gene,mask_gene)
        #[B, 5583] -> [B, 5584, 64]
        peak = self.peak_encoder(peak,mask_peak)
        
        if mask_gene is not None:
        
            m = mask_gene.unsqueeze(-1).float()
            gene = (gene * m).sum(1) / m.sum(1)  
        if mask_peak is not None:
            m = mask_peak.unsqueeze(-1).float()
            peak = (peak * m).sum(1) / m.sum(1)
        x = torch.cat((gene, peak), dim=1)
        # [B, 64] -> [B, 1] -> [B]
        x = self.decoder(x).squeeze()
        return x

    def training_step(self, batch, batch_idx):

        gene, peak,mask,label= batch
        label_stage = label.view(-1)

        pred_stage = self(gene, peak,mask)
        label_stage=label_stage.squeeze(dim=-1)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(pred_stage, label_stage)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        gene, peak,mask,label= batch
        
        label_stage = label.view(-1)

        pred_stage = self(gene, peak,mask)
        label_stage=label_stage.squeeze(dim=-1)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(pred_stage, label_stage)

        self.log('val_loss', loss)

        return loss
    
    def predict_step(self, batch, batch_idx):
        gene, peak,mask,label= batch
        label_stage = label.unsqueeze(1).float()
        pred_stage = self(gene, peak,mask)


        return label_stage, pred_stage
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': step_lr_scheduler,
                'monitor': 'val_loss',
            }
        }
        return optim_dict





device='cuda:0'
import pickle
from tqdm import tqdm
from captum.attr import IntegratedGradients
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
dict_output={}
##################
sample_list=hds_test+inrs_test
###################
dict_sample_cell=load_data('./data/dict_sample_cell.pkl')
with torch.cuda.amp.autocast():
    for test_sample in sample_list:
        idx_list=dict_sample_cell[test_sample]
        dataset_test=scDataset(idx_list,mode)
        
        test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=8)
        model = HIVModel.load_from_checkpoint("./model/hiv_model-epoch=37-val_loss=0.2243.ckpt",map_location='cpu')
        model=model.to(device)
        ig = IntegratedGradients(model)
        dict_sample={'gene':[],'peak':[]}
        for i in tqdm(test_loader): 
            gene, peak,mask,label=i
            gene = gene.to(device)
            peak = peak.to(device)
            mask = mask.to(device)
            attributions = ig.attribute((gene,peak), additional_forward_args=mask,n_steps=50,baselines=(0.0,0.0))
            dict_sample['gene'].append(attributions[0].cpu().detach().numpy())
            dict_sample['peak'].append(attributions[1].cpu().detach().numpy())
            del peak,mask,attributions
        del model
        save_data(dict_sample,'/home/share/huadjyin/home/zhouxuanchi/HIV/final_chance/HDs_INRs_IG/output/'+str(test_sample)+'.pkl')
        

    
