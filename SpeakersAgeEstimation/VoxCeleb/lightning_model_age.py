import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.use_deterministic_algorithms(True)

import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import Accuracy

import pandas as pd
import numpy as np
import wavencoder
import torch.optim as optim
from Model.models import *


class TopME(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.mae = MAE()
        self.mse = MSE()
        self.n = n

    def forward(self, yhat, y):
        abs_diff = torch.abs(yhat-y)
        abs_diff_sort = abs_diff.sort(descending=True)
        abs_diff_top_idx = abs_diff_sort.indices[:self.n].tolist()
        wc_y = y[abs_diff_top_idx]
        wc_yhat = yhat[abs_diff_top_idx]
        return self.mse(wc_yhat, wc_y)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {'SpectralCNNLSTMAge' : SpectralCNNLSTMAge}
        self.model = self.models[HPARAMS['model_type']](HPARAMS['hidden_size'])

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.top_n_me = TopME(n=16)
        self.mae_criterion = MAE()

        self.wdecay = HPARAMS['wdecay']
        self.lr = HPARAMS['lr']

        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path)
        self.a_mean = self.df['Age'].mean()
        self.a_std = self.df['Age'].std()

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_a, y_g, _ = batch
        y_hat_a = self(x)
        y_a = y_a.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_loss = self.regression_criterion(y_hat_a, y_a)
        # age_loss = self.top_n_me(y_hat_a, y_a)

        age_mae = self.mae_criterion(y_hat_a, y_a)

        return {'loss':age_loss,
                'train_age_mae':age_mae.item(),
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch

        self.log('train/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('Train MAE', age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_a, y_g, _ = batch
        y_hat_a = self(x)
        y_a = y_a.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_loss = self.regression_criterion(y_hat_a, y_a)
        # age_loss = self.top_n_me(y_hat_a, y_a)

        age_mae = self.mae_criterion(y_hat_a, y_a)

        return {
                'val_loss':age_loss,
                'val_age_mae':age_mae.item()
                }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch

        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Val MAE', age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_a, y_g, _ = batch
        y_hat_a = self(x)
        y_a = y_a.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_mae = self.mae_criterion(y_hat_a, y_a)
        age_rmse = self.rmse_criterion(y_hat_a, y_a)

        return {
                'age_mae': age_mae.item(),
                'age_rmse': age_rmse.item()
                }

    def test_epoch_end(self, outputs):

        age_mae = torch.tensor([x for x in outputs]).mean()
        age_rmse = torch.tensor([x for x in outputs]).mean()

        pbar = {'age_mae':age_mae.item(),
                'age_rmse':age_rmse.item()
                }
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)