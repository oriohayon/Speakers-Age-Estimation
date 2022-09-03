from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder

N_MFCC = 30

class Ages:
    Range = [10, 20, 40, 60, 80, 100]
    Label = ['10-20', '20-40', '40-60', '60-80', '80-100']
    N_Class = len(Range) - 1
    Mid_Age = [15, 30, 50, 70, 90]


def AgesToRangeIdx(ages_lst):
    ranges_lst = []
    for curr_age in ages_lst:
        for idx, lower_age in enumerate(Ages.Range[:-1]):
            uppper_age = Ages.Range[idx+1]
            if lower_age <= curr_age < uppper_age:
                ranges_lst.append(idx)
    return ranges_lst

class VoxCelebDataset(Dataset):
    def __init__(self,
    wav_folder,
    hparams,
    is_train=True
    ):
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train
        self.wav_len = hparams.wav_len
        self.noise_dataset_path = hparams.noise_dataset_path
        self.data_type = hparams.data_type

        self.speaker_list = self.df.loc[:, 'Unique_ID'].values.tolist()
        self.df.set_index('Unique_ID', inplace=True)
        self.gender_dict = {'male': 0, 'female': 1}

        if self.noise_dataset_path:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random'),
                wavencoder.transforms.AdditiveNoise(self.noise_dataset_path, p=0.5),
                wavencoder.transforms.Clipping(p=0.5),
                ])
        else:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='random'),
                wavencoder.transforms.Clipping(p=0.5),
                ])

        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='center')
            ])

        if 'spectral' in self.data_type:
            # self.spectral_transform = torchaudio.transforms.MelSpectrogram(normalized=True)
            self.spectral_transform = torchaudio.transforms.MFCC(n_mfcc=N_MFCC, log_mels=True)
            self.spec_aug = wavencoder.transforms.Compose([  
                torchaudio.transforms.FrequencyMasking(5),
                torchaudio.transforms.TimeMasking(5),
            ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]

        unique_id = file.split('.')[0]
        if self.df.loc[unique_id, 'Gender'] not in ['male', 'female']:
            print('Unique ID: %s Gender is Unknown' % unique_id)

        gender = self.gender_dict[self.df.loc[unique_id, 'Gender']]
        age = self.df.loc[unique_id, 'Age']

        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))
        if self.is_train:
            wav = self.train_transform(wav)  
            if 'aug' in self.data_type:
                wav = self.spectral_transform(wav)
                wav = self.spec_aug(wav)
                # print('\n Wav file: ')
                # print(list(wav.size()))
            else:
                wav = self.spectral_transform(wav)
        else:
            if 'spectral' in self.data_type:
                wav = self.spectral_transform(wav)

        return wav, age, gender, unique_id
