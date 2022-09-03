import os


class VoxCelebConfig(object):
    # path to the unzuipped VoxCeleb data folder
    data_set  = 'FULL'
    data_path = '../voxceleb_dataset/' + data_set

    # path to csv file containing age, heights of voxceleb speakers
    speaker_csv_path = data_path + '/merged_voxceleb_info.csv'

    # length of wav files for training and testing
    wav_len = 5 * 16000
    # 16000 * 2

    batch_size = 64
    epochs = 500

    # loss = alpha * gender_loss + beta * age_loss
    alpha = 5
    beta = 1

    # data type - raw/spectral/spectral_aug
    data_type = 'spectral_aug'

    # model type
    ## AHG
    # AG: SpectralCNNLSTM
    # A:  SpectralCNNLSTMAge
    # AR:  SpectralCNNLSTMAgeRange / X_vector
    model_type = 'SpectralCNNLSTM'

    # training type - AG - Age&Gender / A - Age / AR - Age Range
    type_dict = {'SpectralCNNLSTM': 'AG', 'SpectralCNNLSTMAge': 'A',
                 'SpectralCNNLSTMAgeRange': 'AR', 'X_vector': 'AR'}
    training_type = type_dict[model_type]

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # Num of GPUs for training and num of workers for datalaoders
    gpu = 0
    n_workers = 12

    # model checkpoint to continue from
    load_model_to_train = False
    model_checkpoint = None #'Final_Models/' + model_type + '_' + data_set + '_DB/model.ckpt'

    # noise dataset for augmentation
    noise_dataset_path = '../noise_dataset'

    # LR of optimizer
    lr = 1e-3

    # Weight decay
    wdecay = 1e-3

    # Test csv output
    test_output_csv_dir = '../Results/' + model_type + '/' + data_set

    # Checkpoints directory
    checkpoint_dir = None

    run_name = 'voxceleb' + data_type + '_' + training_type + '_' + model_type


class TIMITConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '../dataset/wav_data'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = 'Dataset/data_info_height_age.csv'

    # length of wav files for training and testing
    timit_wav_len = 3 * 16000
    # 16000 * 2

    batch_size = 150
    epochs = 200
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 0.1
    beta = 1
    gamma = 0.7

    # training type - AHG/H
    training_type = 'AHG'

    # data type - raw/spectral
    data_type = 'spectral' 

    # model type
    ## AHG 
    # wav2vecLSTMAttn/spectralCNNLSTM/MultiScale
    
    ## H
    # wav2vecLSTMAttn/MultiScale/LSTMAttn
    model_type = 'spectralCNNLSTMAge'

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = '0'
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None
    
    # noise dataset for augmentation
    noise_dataset_path = '/home/shangeth/noise_dataset'

    # LR of optimizer
    lr = 1e-3

    run_name = data_type + '_' + training_type + '_' + model_type