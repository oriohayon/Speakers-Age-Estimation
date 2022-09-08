# Speaker's Age Estimation

This Repository contains the implementation of models, datasets and ideas in the DL final project - "Speaker Age Estimation – “In the Wild”" by Ori Ohayon

## Usage

### Download the dataset
#### VoxCeleb Dataset
We only need the VoxCeleb2 dataset in order to recreate results in our repository

In order to get the VoxCeleb dataset our work used the VoxCelebTrainer Git only for downloading and converting the original full dataset:
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html


### Prepare the dataset for training and testing
In order to create the different datasets out of the original VoxCeleb dataset, we used the create_voxceleb_database.py script and the Enriched_VoxCeleb_dataframe.csv dataframe that contains the detailed information (including the age generated in [1])

Note: 
- The create_voxceleb_database.py needs modifications to create the different datasets

- We provide the csv files contains the information regarding each dataset under the voxceleb_dataset directory

### Training and Testing
In order to train we can run the train_voxceleb.py script with arguments (specified in the config.py file)

To test we can similarly run the test_voxceleb.py script with arguments as in the train (specifying the model to load using --model_checkpoint=<`path to model'>)

We provide the colab notebook SpeakersAgeEst.ipynb with short scripts example of training and testing

## Reference
- [1] Hechmi, Khaled, et al. "VoxCeleb Enrichment for Age and Gender Recognition." arXiv preprint arXiv:2109.13510 (2021).‏
- [2] https://github.com/shangeth/SpeakerProfiling
