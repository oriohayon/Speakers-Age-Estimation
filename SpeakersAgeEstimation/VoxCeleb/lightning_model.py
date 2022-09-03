import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.use_deterministic_algorithms(True)

import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import Accuracy

import pandas as pd
import wavencoder
import torch_optimizer as optim


from Model.models import SpectralCNNLSTM


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
        self.models = {
            'SpectralCNNLSTM': SpectralCNNLSTM
        }
        self.model = self.models[HPARAMS['model_type']](HPARAMS['hidden_size'])

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = HPARAMS['alpha']
        self.beta = HPARAMS['beta']

        self.lr = HPARAMS['lr']
        self.wdecay = HPARAMS['wdecay']

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
        optimizer = optim.DiffGrad(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_a, y_g, _ = batch
        y_hat_a, y_hat_g = self(x)
        y_a, y_g = y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a, y_hat_g = y_hat_a.view(-1).float(), y_hat_g.view(-1).float()
        y_hat_a = y_hat_a # *self.a_std + self.a_mean

        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.beta * age_loss + self.alpha * gender_loss

        age_mae = self.mae_criterion(y_hat_a, y_a)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {'loss':loss, 
                'train_age_mae':age_mae.item(),
                'train_gender_acc':gender_acc,
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['train_gender_acc'] for x in outputs]).mean()

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/a',age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/g',gender_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_a, y_g, _ = batch
        y_hat_a, y_hat_g = self(x)
        y_a, y_g = y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a, y_hat_g = y_hat_a.view(-1).float(), y_hat_g.view(-1).float()
        y_hat_a = y_hat_a # *self.a_std + self.a_mean

        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.alpha * gender_loss + self.beta * age_loss

        age_mae = self.mae_criterion(y_hat_a, y_a)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {'val_loss':loss, 
                'val_age_mae':age_mae.item(),
                'val_gender_acc':gender_acc}

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['val_gender_acc'] for x in outputs]).mean()
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/a',age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/g',gender_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_a, y_g = batch
        y_hat_a, y_hat_g = self(x)
        y_a, y_g = y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a, y_hat_g = y_hat_a.view(-1).float(), y_hat_g.view(-1).float()
        y_hat_a = y_hat_a # *self.a_std + self.a_mean

        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        idx = y_g.view(-1).long()
        female_idx = torch.nonzero(idx).view(-1)
        male_idx = torch.nonzero(1-idx).view(-1)

        male_age_mae = self.mae_criterion(y_hat_a[male_idx], y_a[male_idx])

        female_age_mae = self.mae_criterion(y_hat_a[female_idx], y_a[female_idx])

        male_age_rmse = self.rmse_criterion(y_hat_a[male_idx], y_a[male_idx])

        female_age_rmse = self.rmse_criterion(y_hat_a[female_idx], y_a[female_idx])

        return {
                'male_age_mae':male_age_mae.item(),
                'female_age_mae':female_age_mae.item(),
                'male_age_rmse':male_age_rmse.item(),
                'female_age_rmse':female_age_rmse.item(),
                'test_gender_acc':gender_acc
        }

    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        male_age_mae = torch.tensor([x['male_age_mae'] for x in outputs]).mean()
        female_age_mae = torch.tensor([x['female_age_mae'] for x in outputs]).mean()

        male_age_rmse = torch.tensor([x['male_age_rmse'] for x in outputs]).mean()
        female_age_rmse = torch.tensor([x['female_age_rmse'] for x in outputs]).mean()

        gender_acc = torch.tensor([x['test_gender_acc'] for x in outputs]).mean()

        pbar = {
            'male_age_mae':male_age_mae.item(),
            'female_age_mae': female_age_mae.item(),
            'male_age_rmse':male_age_rmse.item(),
            'female_age_rmse': female_age_rmse.item(),
            'test_gender_acc':gender_acc.item()
        }
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)