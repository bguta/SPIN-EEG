# Reproducing experiments

### Train autoencoder w/ 5 subjects, 12 features and 100ms windows
```console
foo@bar:~$ python3 autoencoder_eeg.py --num_subjects 5 --time_window 100 --encoder_features 12 --eeg_dir ../ALIGNED_DATA --data_split_seed 2754047194 --subject_seed 2584610193
```

### 5 class classifier with autoencoder
```console
foo@bar:~$ python3 classifier_emotion_state.py --num_subjects 5 --time_window 100 --classifier_classes 5 --eeg_dir ../ALIGNED_DATA --data_split_seed 2754047194 --subject_seed 2584610193 --encoder_features 12 --encoder_path results/1642721614140_encoder_model.pth
```

### 3 class classifier with no autoencoder
```console
foo@bar:~$ python3 classifier_emotion_state.py --training_save_path 'experiment_id_001.pth' --data_split_seed 128 --subject_seed 55 --num_subjects 1 --time_window 100
```

### custom experiment with all parameters
```console
foo@bar:~$ python3 classifier_emotion_state.py --eeg_dir '../Aligned' --time_window 100 --data_split_seed 128 --num_subjects 3 --subject_seed 55
                                       --classifier_hidden_size 32 --classifier_classes 10 --classifier_dropout 0.5 --training_epochs 10
                                       --training_batch_size 128 --training_lr 0.005 --training_resume_path experiment_id_001.pth --training_save_path experiment_id_001.pth
```
