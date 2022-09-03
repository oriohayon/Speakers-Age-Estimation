import csv

from config import VoxCelebConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os

from VoxCeleb.dataset import VoxCelebDataset
from VoxCeleb.lightning_model import LightningModel
from VoxCeleb.lightning_model_age import LightningModel as LightningModelAge
from VoxCeleb.lightning_model_age_range import LightningModel as LightningModelAgeRange

from VoxCeleb.dataset import Ages, AgesToRangeIdx

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl


import torch
import torch.utils.data as data

from tqdm import tqdm 
import pandas as pd
import numpy as np

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=VoxCelebConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=VoxCelebConfig.speaker_csv_path)
    parser.add_argument('--wav_len', type=int, default=VoxCelebConfig.wav_len)
    parser.add_argument('--batch_size', type=int, default=VoxCelebConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=VoxCelebConfig.epochs)
    parser.add_argument('--alpha', type=float, default=VoxCelebConfig.alpha)
    parser.add_argument('--beta', type=float, default=VoxCelebConfig.beta)
    parser.add_argument('--wdecay', type=float, default=VoxCelebConfig.wdecay)
    parser.add_argument('--hidden_size', type=float, default=VoxCelebConfig.hidden_size)
    parser.add_argument('--lr', type=float, default=VoxCelebConfig.lr)
    parser.add_argument('--gpu', type=int, default=VoxCelebConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=VoxCelebConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=VoxCelebConfig.model_checkpoint)
    parser.add_argument('--noise_dataset_path', type=str, default=VoxCelebConfig.noise_dataset_path)
    parser.add_argument('--test_output_csv_dir', type=str, default=VoxCelebConfig.test_output_csv_dir)
    parser.add_argument('--model_type', type=str, default=VoxCelebConfig.model_type)
    parser.add_argument('--training_type', type=str, default=VoxCelebConfig.training_type)
    parser.add_argument('--data_type', type=str, default=VoxCelebConfig.data_type)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print('Testing ' + hparams.model_type + ' from ' + hparams.model_checkpoint + ' on VoxCeleb Dataset')
    print(f'Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Testing Dataset
    test_set = VoxCelebDataset(
        wav_folder = os.path.join(hparams.data_path, 'TEST'),
        hparams = hparams,
        is_train=False
        )

    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    a_mean = df['Age'].mean()
    a_std = df['Age'].std()

    #Testing the Model
    if hparams.model_checkpoint:
        if hparams.training_type == 'AG':
            model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
            model.eval()
            age_pred = []
            age_true = []
            gender_pred_tensors = []
            gender_true = []
            unique_ids = []

            # Create output csv path
            if not os.path.exists(hparams.test_output_csv_dir):
                os.makedirs(hparams.test_output_csv_dir)

            # i = 0
            for batch in tqdm(test_set):
                x, y_a, y_g, unique_id = batch
                y_hat_a, y_hat_g = model(x)

                age_pred.append((y_hat_a).item())
                gender_pred_tensors.append(y_hat_g>0.5)

                age_true.append(y_a.item())
                gender_true.append(y_g)
                unique_ids.append(unique_id)

            # Convert gender predict to logical values
            gender_pred = [int(x.item()) for x in gender_pred_tensors]

            female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
            male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

            age_true = np.array(age_true)
            age_pred = np.array(age_pred)

            output_csv_path = hparams.test_output_csv_dir + '/output_test_AG.csv'
            with open(output_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Unique ID', 'Real Gender', 'Estimated Gender', 'Real Age', 'Estimated Age'])
                for i in range(len(age_true)):
                    gender_real = 'female' if gender_true[i] else 'male'
                    gender_est  = 'female' if gender_pred[i] else 'male'
                    writer.writerow([unique_ids[i], gender_real, gender_est, age_true[i], age_pred[i]])

            amae = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
            armse = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
            print('Male Age RMSE: %.4f , Age MAE: %.4f ' % (armse, amae))

            amae = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
            armse = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
            print('Female Age RMSE: %.4f , Age MAE: %.4f ' % (armse, amae))

            amae = mean_absolute_error(age_true, age_pred)
            armse = mean_squared_error(age_true, age_pred, squared=False)
            print('Total RMSE: %.4f , Total MAE: %.4f' % (armse, amae))

            print('Gender Accuracy: %.4f' % accuracy_score(gender_true, gender_pred))

        elif hparams.training_type == 'A':
            model = LightningModelAge.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
            model.eval()
            age_pred = []
            age_true = []
            unique_ids = []

            # Create output csv path
            if not os.path.exists(hparams.test_output_csv_dir):
                os.makedirs(hparams.test_output_csv_dir)

            for batch in tqdm(test_set):
                x, y_a, y_g, unique_id = batch
                y_hat_a = model(x)

                age_pred.append((y_hat_a).item())
                age_true.append(y_a.item())
                unique_ids.append(unique_id)
                # print('Unique ID: %s' % unique_id)

            age_true = np.array(age_true)
            age_pred = np.array(age_pred)

            amae = mean_absolute_error(age_true, age_pred)
            armse = mean_squared_error(age_true, age_pred, squared=False)

            output_csv_path = hparams.test_output_csv_dir + '/output_test_A.csv'
            with open(output_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Unique ID', 'Real Age', 'Estimated Age'])
                for i in range(len(age_true)):
                    writer.writerow([unique_ids[i], age_true[i], age_pred[i]])

            print('Age RMSE: %.4f , Age MAE: %.4f ' % (armse, amae))

        elif hparams.training_type == 'AR':
            model = LightningModelAgeRange.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
            model.eval()
            age_range_pred = []
            age_range_true = []
            ages_true = []
            unique_ids = []
            gender_real = []

            # Create output csv path
            if not os.path.exists(hparams.test_output_csv_dir):
                os.makedirs(hparams.test_output_csv_dir)

            for batch in tqdm(test_set):
                x, y_a, y_g, unique_id = batch
                range_est = model(x)
                if len(range_est.size()) > 1:
                    est_labels = range_est.argmax(dim=1)
                else:
                    est_labels = range_est.argmax(dim=0)
                age_range_pred.append(est_labels.item())
                ages_true.append(y_a.item())
                unique_ids.append(unique_id)
                gender_real.append('female' if y_g else 'male')
                # print('Unique ID: %s' % unique_id)

            age_range_true = AgesToRangeIdx(ages_true)
            age_range_true = [int(x) for x in age_range_true]
            print('Age Range index True: ')
            print(age_range_true)
            age_range_true_labels = [Ages.Label[x] for x in age_range_true]
            age_range_pred = [int(x) for x in age_range_pred]
            print('Age Range index Predict: ')
            print(age_range_pred)
            age_range_pred_labels = [Ages.Label[x] for x in age_range_pred]

            compare_lst = [age_range_true[idx]==age_range_pred[idx] for idx in range(len(age_range_true))]
            print(compare_lst)

            accuracy = float(sum(compare_lst)) / len(age_range_true)

            output_csv_path = hparams.test_output_csv_dir + '/output_test_AR.csv'
            with open(output_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Unique ID', 'Gender', 'Real Age', 'Estimated Age'])
                for i in range(len(age_range_pred_labels)):
                    writer.writerow([unique_ids[i], gender_real[i], age_range_true_labels[i], age_range_pred_labels[i]])

            print('Age Range Accuracy: %.4f' % accuracy)

        else:
            raise ValueError('Bad training type in VoxCeleb Configs')
    else:
        print('Model chekpoint not found for Testing !!!')
