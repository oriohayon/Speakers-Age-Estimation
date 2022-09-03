import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.autograd import Variable
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import Accuracy

import pandas as pd
import numpy as np
import wavencoder
import torch.optim as optim
from Model.models import *
from VoxCeleb.dataset import Ages, AgesToRangeIdx


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


def AgeToRangeLabel(age_tensor):
    range_labels = np.zeros((len(age_tensor)))
    for idx in range(len(age_tensor)):
        for range_idx in range(len(Ages.Range)-1):
            if Ages.Range[range_idx] <= age_tensor[idx] < Ages.Range[range_idx+1]:
                range_labels[idx] = range_idx
    
    return torch.tensor(range_labels, dtype=torch.long)


class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {'SpectralCNNLSTMAgeRange' : SpectralCNNLSTMAgeRange,
                       'X_vector': X_vector}
        self.model = self.models[HPARAMS['model_type']](HPARAMS['hidden_size'])

        self.CrEntloss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.mse = MSE()
        self.rmse = RMSELoss()
        self.wdecay = HPARAMS['wdecay']
        self.lr = HPARAMS['lr']
        self.alpha = HPARAMS['alpha']

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
        x, y_a, y_g, unique_id = batch
        # print('batch ages: ')
        # print(y_a)
        # print('Labels tensor: ')
        range_targets = torch.tensor(AgesToRangeIdx(y_a), dtype=torch.long).to(device)
        ages_targets = y_a
        # print('Labels targets:')
        # print(range_targets.size())
        range_est = self(x)
        # print('Output of CNN:')
        # print(range_est)

        if len(range_est.size()) > 1:
            est_labels = range_est.argmax(dim=1)
        else:
            est_labels = range_est.argmax(dim=0)
        est_labels.to(device)
        # est_ages = torch.tensor([Ages.Mid_Age[range_idx] for range_idx in est_labels], dtype=torch.float64).to(device)
        # print(est_labels.view(-1))

        loss = self.CrEntloss(range_est, range_targets.squeeze(0))
        # print('Cross entropy loss:')
        # print(cross_ent_loss)
        # loss = Variable(self.rmse(est_ages, ages_targets) * self.alpha, requires_grad=True)
        train_accuracy   = self.accuracy(est_labels.view(-1).long(), range_targets.view(-1).long())

        # print('Entropy loss: %f' % cross_ent_loss)
        # print('Accuracy: %f' % train_accuracy)

        return {
                'loss': loss,
                'train_acc': train_accuracy
               }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        train_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        train_acc = torch.tensor([x['train_acc'] for x in outputs]).mean()

        self.log('train/loss', train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', train_acc.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_a, y_g, unique_id = batch
        # print('Truth age Range: ')
        # print(AgesToRangeIdx(y_a))
        range_targets = torch.tensor(AgesToRangeIdx(y_a), dtype=torch.long).to(device)
        ages_targets = y_a
        range_est = self(x)
        # print('Output of CNN:')
        # print(range_est)

        # raise ValueError('Break')

        if len(range_est.size()) > 1:
            est_labels = range_est.argmax(dim=1)
        else:
            est_labels = range_est.argmax(dim=0)
        est_labels.to(device)
        # est_ages = torch.tensor([Ages.Mid_Age[range_idx] for range_idx in est_labels], dtype=torch.float64).to(device)
        
        loss = self.CrEntloss(range_est, range_targets)
        # loss = Variable(self.rmse(est_ages, ages_targets), requires_grad=False)
        val_accuracy   = self.accuracy(est_labels.view(-1).long(), range_targets.view(-1).long())
        # print('Validation accuracy: ')
        # print(val_accuracy)
        return {
                'val_loss': loss,
                'val_acc': val_accuracy
               }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()

        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Val/acc', val_acc.item(), on_step=False, on_epoch=True, prog_bar=True)