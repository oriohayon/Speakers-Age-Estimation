import torch
import torch.nn as nn
import wavencoder
from blitz.modules import BayesianLSTM
from VoxCeleb.dataset import Ages
from Model.utils import TDNN
from VoxCeleb.dataset import N_MFCC
import torch.nn.functional as F
from VoxCeleb.dataset import Ages

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class X_vector(nn.Module):
    def __init__(self, hidden_size, input_dim=N_MFCC, num_classes=Ages.N_Class, p_dropout=0.25):
        super(X_vector, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_size, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(hidden_size) #, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=0.25)

        self.tdnn2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(hidden_size) #, momentum=0.1) #, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=0.25)

        self.tdnn3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(hidden_size) # , momentum=0.1) #, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=0.25)

        self.tdnn4 = nn.Conv1d(in_channels=hidden_size, out_channels=2*hidden_size, kernel_size=2, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(2*hidden_size) # , momentum=0.1) #, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=0.25)

        self.tdnn5 = nn.Conv1d(in_channels=2*hidden_size, out_channels=2*hidden_size, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(2*hidden_size) #  , momentum=0.1) #, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(4*hidden_size, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size) #, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(hidden_size, int(hidden_size/4))
        self.bn_fc2 = nn.BatchNorm1d(int(hidden_size/4)) #, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(int(hidden_size/4), num_classes)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = x.squeeze(1)

        # x = x.transpose(1,2)
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
            
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.soft_max(self.fc3(x))

        return x


class SpectralCNNLSTMAgeRange(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(N_MFCC, int(hidden_size), kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size)),
            nn.Conv1d(int(hidden_size), int(hidden_size), kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size)),
            nn.Conv1d(int(hidden_size), int(hidden_size), kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size))
        )
        self.lstm = nn.LSTM(int(hidden_size), int(hidden_size), batch_first=True)
        # self.lstm = BayesianLSTM(int(lstm / 2), int(lstm / 2))
        self.attention = wavencoder.layers.SoftAttention(int(hidden_size), int(hidden_size))

        self.age_regressor = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size/2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/2), int(hidden_size/2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/2), Ages.N_Class))

        self.soft_max_0 = nn.Softmax(dim=0)
        self.soft_max_1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.squeeze(1)
        # print('Input size after Squeeze')
        # print(list(x.size()))
        x = self.encoder(x)
        # print('Input size after Encoder')
        # print(list(x.size()))
        output, (hidden, _) = self.lstm(x.transpose(1, 2))
        # print('lstm output:')
        # print(output.size())
        attn_output = self.attention(output)
        # print('attention output: ')
        # print(list(attn_output.size()))
        range_probs_att = self.age_regressor(attn_output)
        if len(range_probs_att.size()) > 1:
          range_probs = self.soft_max_1(range_probs_att)
        else:
          range_probs = torch.tensor([list(self.soft_max_0(range_probs_att))]).to(device)
        # print('Age:')
        # print(list(age.size()))

        return range_probs


class SpectralCNNLSTMAge(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(N_MFCC, int(hidden_size), kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size)),
            nn.Conv1d(int(hidden_size), int(hidden_size), kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size)),
            nn.Conv1d(int(hidden_size), int(hidden_size), kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size))
        )
        self.lstm = nn.LSTM(int(hidden_size), int(hidden_size), batch_first=True)
        # self.lstm = BayesianLSTM(int(lstm / 2), int(lstm / 2))
        self.attention = wavencoder.layers.SoftAttention(int(hidden_size), int(hidden_size))

        self.age_regressor = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size/2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/4), 1))

    def forward(self, x):
        x = x.squeeze(1)
        # print('Input size after Squeeze')
        # print(list(x.size()))
        x = self.encoder(x)
        # print('Input size after Encoder')
        # print(list(x.size()))
        output, (hidden, _) = self.lstm(x.transpose(1, 2))
        # print('lstm output:')
        # print(output.size())
        attn_output = self.attention(output)
        # print('attention output: ')
        # print(list(attn_output.size()))
        age = self.age_regressor(attn_output)
        # print('Age:')
        # print(list(age.size()))

        return age


class SpectralCNNLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(N_MFCC, int(hidden_size/2), kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2)),
            nn.Conv1d(int(hidden_size/2), int(hidden_size/2), kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2)),
            nn.Conv1d(int(hidden_size/2), int(hidden_size/2), kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden_size/2))
        )
        self.lstm_1 = nn.LSTM(int(hidden_size / 2), int(hidden_size / 2), batch_first=True)
        self.lstm_2 = nn.LSTM(int(hidden_size / 2), int(hidden_size / 2), batch_first=True)
        # self.lstm = BayesianLSTM(int(lstm / 2), int(lstm / 2))
        self.attention_1 = wavencoder.layers.SoftAttention(int(hidden_size / 2), int(hidden_size / 2))
        self.attention_2 = wavencoder.layers.SoftAttention(int(hidden_size / 2), int(hidden_size / 2))

        self.age_regressor = nn.Sequential(
            nn.Linear(int(hidden_size/2), int(hidden_size/2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/4), 1))
        self.gender_classifier = nn.Sequential(
            nn.Linear(int(hidden_size / 2), int(hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size / 2), int(hidden_size/4)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hidden_size/4), 1))

    def forward(self, x):
        x = x.squeeze(1)
        x = self.encoder(x)
        output_1, (hidden_1, _) = self.lstm_1(x.transpose(1, 2))
        output_2, (hidden_2, _) = self.lstm_2(x.transpose(1, 2))
        attn_output_1 = self.attention_1(output_1)
        attn_output_2 = self.attention_2(output_2)
        # print('attention output: ')
        # print(list(attn_output.size()))
        age = self.age_regressor(attn_output_1)
        # print('Age:')
        # print(list(age.size()))
        gender = self.gender_classifier(attn_output_2)
        # print('Gender:')
        # print(list(gender.size()))
        return age, gender
