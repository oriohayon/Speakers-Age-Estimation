import csv
import glob
import os
import subprocess
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_csv_as_dict(csv_path):
    columns = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                # first row
                columns = [[value] for value in row]
    # you now have a column-major 2D array of your file.
    dataset_dict = {c[0]: c[1:] for c in columns}
    return dataset_dict


def save_as_csv(dict, csv_path):
    new_header = list(dict.keys())
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        for idx, voxceleb_id in enumerate(dict['VoxCeleb_ID']):
            tmp_row = []
            for key in dict.keys():
                tmp_row.append(dict[key][idx])
            writer.writerow(tmp_row)


def create_enriched_csv_dataset(input_csv_file_path, output_csv_file_path, single_video_flag=False):
    data_dict = {'Name': [], 'Gender': [], 'Nationality': [], 'VoxCeleb_ID': [], 'Video_ID': [], 'Video_Title': [], 'Age': []}
    done_ids  = []
    # Create full dataframe dictionary
    with open(input_csv_file_path, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in tqdm(reader):
            # Ignore all data without age or gender
            if row['speaker_age'] != "" and row['gender'].lower() in ['male', 'female']:
                if row['VoxCeleb_ID'] not in done_ids:
                    # Simplify dictionary for output
                    data_dict['Name'].append(row['Name'])
                    data_dict['Age'].append(row['speaker_age'])
                    data_dict['VoxCeleb_ID'].append(row['VoxCeleb_ID'])
                    data_dict['Gender'].append(row['gender'])
                    data_dict['Video_ID'].append(row['video_id'])
                    data_dict['Video_Title'].append(row['title'])
                    data_dict['Nationality'].append(row['nationality_wiki'])
                    if single_video_flag:
                        done_ids.append(row['VoxCeleb_ID'])

    # Create new narrowed CSV:
    save_as_csv(data_dict, output_csv_file_path)

def create_relevant_wav_dataset(input_dataset_path, target_dataset_path, input_csv_path, target_csv_path,
                                single_rec_flag=True, use='TRN', gender=None, single_id_flag=False, max_hist=200):
    """
    In this function we assume that the input dataset is wav and given in the following order:
    <input_dataset_path> / <VoxCeleb_ID> / <Video_ID> / <sample_index>.wav
    Target: copy only relevant data to the full_data_set_path
    """
    # Get target dataset information from csv
    dataset_dict = read_csv_as_dict(input_csv_path)
    relevant_dataset_dict = {x: [] for x in dataset_dict.keys()}
    relevant_dataset_dict['Unique_ID'] = []
    relevant_dataset_dict['index'] = []
    relevant_dataset_dict['Use'] = []
    relevant_idx = 0
    done_ids = []
    age_histogram = np.zeros(9)

    # Create dataset target directory if not exists
    if not os.path.exists(target_dataset_path):
        os.makedirs(target_dataset_path)

    dataset_ids = os.listdir(input_dataset_path)
    num_of_bad_ids = 0
    for idx, id in enumerate(tqdm(dataset_dict['VoxCeleb_ID'])):
        if (gender is not None and gender.lower() != dataset_dict['Gender'][idx]) or (single_id_flag and id in done_ids):
            continue
        age_idx = int(np.floor(float(dataset_dict['Age'][idx])/10 - 1))
        if age_histogram[age_idx] > max_hist:
            continue
        if id in dataset_ids:
            tmp_path = input_dataset_path + '/' + id
            videos_id = os.listdir(tmp_path)
            if dataset_dict['Video_ID'][idx] in videos_id:
                video_path = tmp_path + '/' + dataset_dict['Video_ID'][idx]
                rec_lst = os.listdir(video_path)
                if single_rec_flag:
                    rec_lst = [random.choice(rec_lst)]
                for rec in rec_lst:
                    input_file_path = video_path + '/' + rec
                    unique_id = id + '_' + dataset_dict['Video_ID'][idx]
                    target_file_path = target_dataset_path + '/' + unique_id + '.wav'
                    subprocess.call('cp -f "%s" "%s"' % (input_file_path, target_file_path), shell=True)
                    # Update new Dictionary
                    relevant_dataset_dict['Unique_ID'].append(unique_id)
                    relevant_dataset_dict['Name'].append(dataset_dict['Name'][idx])
                    relevant_dataset_dict['Age'].append(dataset_dict['Age'][idx])
                    relevant_dataset_dict['VoxCeleb_ID'].append(dataset_dict['VoxCeleb_ID'][idx])
                    relevant_dataset_dict['Gender'].append(dataset_dict['Gender'][idx])
                    relevant_dataset_dict['Video_ID'].append(dataset_dict['Video_ID'][idx])
                    relevant_dataset_dict['Video_Title'].append(dataset_dict['Video_Title'][idx])
                    relevant_dataset_dict['Nationality'].append(dataset_dict['Nationality'][idx])
                    relevant_dataset_dict['Use'].append(use)
                    relevant_dataset_dict['index'].append(relevant_idx)
                    relevant_idx += 1
                    done_ids.append(dataset_dict['VoxCeleb_ID'][idx])
                    age_histogram[age_idx] += 1
        else:
            num_of_bad_ids += 1

    save_as_csv(relevant_dataset_dict, target_csv_path)


def split_train_val(full_train_dataset_path, val_dataset_path, train_csv_path, val_to_train_ratio=0.1):
    # full_train_dataset_files = os.listdir(full_train_dataset_path)
    # val_dataset_files = random.sample(full_train_dataset_files, round(val_to_train_ratio * len(full_train_dataset_files)))

    # Create validation folder if not exists
    if not os.path.exists(val_dataset_path):
        os.makedirs(val_dataset_path)

    train_dict = read_csv_as_dict(train_csv_path)
    validation_ids = random.sample(train_dict['Unique_ID'], round(val_to_train_ratio*len(train_dict['Unique_ID'])))
    val_idx_lst = [train_dict['Unique_ID'].index(curr_id) for curr_id in validation_ids]
    for i, use in enumerate(train_dict['Use']):
        if i in val_idx_lst:
            train_dict['Use'][i] = 'TST'

    for id in tqdm(validation_ids):
        full_input_path = full_train_dataset_path + '/' + id + '.wav'
        full_target_path = val_dataset_path + '/' + id + '.wav'
        subprocess.call('mv "%s" "%s"' % (full_input_path, full_target_path), shell=True)

    save_as_csv(train_dict, train_csv_path)


def merge_train_test_csvs(train_csv_path, test_csv_path, target_csv_path):
    train_dict = read_csv_as_dict(train_csv_path)
    test_dict = read_csv_as_dict(test_csv_path)
    for tmp_key in train_dict.keys():
        train_dict[tmp_key] = train_dict[tmp_key] + test_dict[tmp_key]
    # Re-index
    for i in range(len(train_dict['index'])):
        train_dict['index'][i] = i
    save_as_csv(train_dict, target_csv_path)


if __name__ == '__main__':
    full_df_csv_file_path   = '/Users/oriohayon/GitHub/voxceleb_enrichment_age_gender/dataset/final_dataframe_extended.csv'
    enriched_csv_file_path  = '/Users/oriohayon/GitHub/voxceleb_enrichment_age_gender/dataset/final_narrowed.csv'
    # create_enriched_csv_dataset(full_df_csv_file_path, enriched_csv_file_path)

    # # Generate Train & Val dataset:
    dataset_name = 'FULL_BAL'
    train_dataset_path      = '/Volumes/GoogleDrive/My Drive/DL_Projects/Final Project/SpeakerProfiling Master/voxceleb_dataset/'+dataset_name+'/TRAIN'
    val_dataset_path        = '/Volumes/GoogleDrive/My Drive/DL_Projects/Final Project/SpeakerProfiling Master/voxceleb_dataset/'+dataset_name+'/VAL'
    test_dataset_path       = '/Volumes/GoogleDrive/My Drive/DL_Projects/Final Project/SpeakerProfiling Master/voxceleb_dataset/'+dataset_name+'/TEST'
    train_target_csv_path   = '/Volumes/GoogleDrive/My Drive/DL_Projects/Final Project/SpeakerProfiling Master/voxceleb_dataset/'+dataset_name+'/train_info.csv'
    test_target_csv_path    = '/Volumes/GoogleDrive/My Drive/DL_Projects/Final Project/SpeakerProfiling Master/voxceleb_dataset/'+dataset_name+'/test_info.csv'
    merged_csv_path         = '/Volumes/GoogleDrive/My Drive/DL_Projects/Final Project/SpeakerProfiling Master/voxceleb_dataset/'+dataset_name+'/merged_voxceleb_info.csv'
    input_dataset_path      = '/Users/oriohayon/GitHub/voxceleb_trainer/data/voxceleb2'
    create_relevant_wav_dataset(input_dataset_path, train_dataset_path, enriched_csv_file_path, train_target_csv_path,
                                use='TRN', single_id_flag=True, max_hist=50)
    split_train_val(train_dataset_path, val_dataset_path, train_target_csv_path)
    # # Generate Test dataset:
    input_dataset_path      = '/Users/oriohayon/GitHub/voxceleb_trainer/data/voxceleb2_test'
    create_relevant_wav_dataset(input_dataset_path, test_dataset_path, enriched_csv_file_path, test_target_csv_path,
                                use='TST')
    merge_train_test_csvs(train_target_csv_path, test_target_csv_path, merged_csv_path)