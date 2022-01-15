# EEG Stress Classification 

This is a project sponsored by the SPIN lab at UBC. The goal of the project is to use the captured EEG and stress response data from previous experiments
to create a state-of-the-art model that can correctly classify the stress from the EEG input.

Unprocessed EEG data is contained int the RAW_EEG folder, while the normalized EEG data is contained in the EEG_FT_DATA folder.

## Autoencoder Feature Generation
![Autoencoder-subject-9](results/subject_9_EEG_channel_1_raw.png)

## Classifying Stress Level [100 ms timewindows; low/med/high]
![stress-level-subject-9](results/subject_9_EEG_encoded_labels_100ms_8_features_3_classes.png)

## Classifying Stress Transition [100 ms timewindows; decrease/no-change/increase]
![stress-transition-subject-9](results/subject_9_EEG_encoded_transitions_100ms_8_features.png)

# Guide
    - Make sure the requirements shown in requirements.txt are satisfied for the installation of python
    - Copy trial_data_split-anon.zip into the RAW_EEG folder and unzip (RAW_EEG/p1 RAW_EEG/p2 etc should be visible)
    - From the src directory run python3 main.py or python main.py on non unix machines. This will create the csv files for each subject and will take some time since the dataset is large (~5GB)
    - The csv files used for training are located in 'ALIGEND_DATA' by default
    - Explore the jupyter notebooks to see how the models are trained
    - From the Experiments directory run the files to reproduce some results