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






#############################################
mode='hds_or_irs'
#############################################
from sklearn.model_selection import train_test_split
from collections import Counter
def get_index_to_train(mode,not_train_list=['HD-H162','HD-H323','HD-H330','HD-H150','HD-H325','PD-H292','PD-H262','PD-H296','PD-H279','PD-H297','PD-H263','PD-H232','PD-H230','PD-H237','PD-H233']):
    dict_sample_stage=load_data('/home/share/huadjyin/home/zhouxuanchi/HIV/new_atac_and_gene_to_class_faster/find_important_again/train_irs_inrs/data/dict_sample_stage.pkl')
    dict_sample_cell=load_data('/home/share/huadjyin/home/zhouxuanchi/HIV/new_atac_and_gene_to_class_faster/find_important_again/train_irs_inrs/data/dict_sample_cell.pkl')
    list_hds=[]
    for i in dict_sample_stage['HDs']:
        if i not in not_train_list:
            list_hds+=dict_sample_cell[i]
    list_inrs=[]
    for i in dict_sample_stage['INRs']:
        if i not in not_train_list:
            list_inrs+=dict_sample_cell[i]
    list_irs=[]
    for i in dict_sample_stage['IRs']:
        if i not in not_train_list:
            list_irs+=dict_sample_cell[i]
    if mode=='hds_or_inrs':
        X=list_hds+list_inrs
        y=[0]*len(list_hds)+[1]*len(list_inrs)
    elif mode=='hds_or_irs':
        X=list_hds+list_irs
        y=[0]*len(list_hds)+[1]*len(list_irs)
    else:
        X=list_irs+list_inrs
        y=[0]*len(list_irs)+[1]*len(list_inrs)        
    return X,y

X,y=get_index_to_train(mode=mode)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
counter = Counter(y_train)
pos_weight=counter[0]/counter[1]



#############################################

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
###########################################
train_dataset = scDataset(X_train,mode)
val_dataset = scDataset(X_test,mode)



class HIVModel_irs_inrs_faster(pl.LightningModule):
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

#################################

early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=5, 
    mode='min'  
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    mode='min',  
    save_top_k=1,  
    dirpath='./model/',  
    filename='hiv_model-{epoch:02d}-{val_loss:.4f}',#-{R:.4f}
    verbose=True,
    save_last=False,
    save_weights_only=True  
)

logger = TensorBoardLogger(save_dir='logs/atac/',name="run")
trainer = pl.Trainer(
    accelerator='gpu',  
    devices=[0,1,2,3],
    max_epochs=100,
    logger=logger,
    precision='bf16-mixed',
    strategy='ddp',
    callbacks=[early_stopping_callback, checkpoint_callback],  # 添加回调函数
)

model = HIVModel_irs_inrs_faster()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, worker_init_fn=seed_worker)

val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, worker_init_fn=seed_worker)

trainer.fit(model, train_loader, val_loader)
