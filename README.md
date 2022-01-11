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
