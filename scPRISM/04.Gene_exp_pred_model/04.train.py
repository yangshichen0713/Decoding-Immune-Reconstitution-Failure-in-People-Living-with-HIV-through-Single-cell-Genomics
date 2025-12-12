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
    def __init__(self,index):
        self.path='./adata_process'
        #self.path='/home/share/huadjyin/home/zhouxuanchi/HIV/atac_to_gene_new_data_0218/adata_process'
        self.index_list=index
    def __len__(self):
        return len(self.index_list)
    def get_np_array(self, filename):
        return np.load(os.path.join(self.path, filename))
    def __getitem__(self, idx):
        index_name=self.index_list[idx]
        array_idx=self.get_np_array(str(index_name)+'.npy')
        gene = torch.tensor(array_idx[:582], dtype=torch.bfloat16)
        peak = torch.tensor(array_idx[582:], dtype=torch.bfloat16)

        mask=torch.tensor((gene != 0), dtype=torch.bfloat16)
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


class Peak2GeneModel(pl.LightningModule):
    def __init__(self, input_dim=64,hidden_dim=512,out_features=582):
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
        return pred_gene
    def training_step(self, batch, batch_idx):
        gene,peak,mask=batch
        pred_gene=self(peak,mask)
        loss = F.mse_loss(pred_gene.view_as(gene), gene)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        gene,peak,mask=batch
        pred_gene=self(peak,mask)
        loss = F.mse_loss(pred_gene.view_as(gene), gene)
        self.log('val_loss', loss)
        return loss
    def predict_step(self, batch, batch_idx):
        gene,peak,mask=batch
        pred_gene=self(peak,mask)
        return gene,pred_gene
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
    


if __name__ == "__main__":
    setup_seed(3407)
    dict_fen=load_data('./data/fen.pkl')
    X_train = dict_fen['train']
    X_val = dict_fen['val']
    
    train_dataset = scDataset(X_train)
    val_dataset = scDataset(X_val)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  
        patience=15, 
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
    model = Peak2GeneModel()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, worker_init_fn=seed_worker)

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, worker_init_fn=seed_worker)

    trainer.fit(model, train_loader, val_loader)
