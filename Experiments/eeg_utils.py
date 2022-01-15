import numpy as np
import os
import glob
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

#################################
#################################
#### Load Dataset
def split_filtered_and_raw(filtered_data : list, raw_data : list, random_seed=128) -> list:
    
    autoencoder_split, classifier_split = train_test_split(filtered_data, test_size=0.7, random_state=random_seed, shuffle=True)
    autoencoder_split_raw, classifier_split_raw = train_test_split(raw_data, test_size=0.7, random_state=random_seed, shuffle=True)
    
    # split data into train and test
    autoencoder_train, autoencoder_test = train_test_split(np.array(autoencoder_split), test_size=0.2, random_state=random_seed, shuffle=True)
    autoencoder_train_raw, autoencoder_test_raw = train_test_split(np.array(autoencoder_split_raw), test_size=0.2, random_state=random_seed, shuffle=True)
    
    classifier_train, classifier_test = train_test_split(np.array(classifier_split), test_size=0.2, random_state=random_seed, shuffle=True)
    classifier_train_raw, classifier_test_raw = train_test_split(np.array(classifier_split_raw), test_size=0.2, random_state=random_seed, shuffle=True)

    
    # further split train into validation and train
    autoencoder_train, autoencoder_val = train_test_split(autoencoder_train, test_size=0.2, random_state=random_seed, shuffle=True)
    autoencoder_train_raw, autoencoder_val_raw = train_test_split(autoencoder_train_raw, test_size=0.2, random_state=random_seed, shuffle=True)
    
    classifier_train, classifier_val = train_test_split(classifier_train, test_size=0.2, random_state=random_seed, shuffle=True)
    classifier_train_raw, classifier_val_raw = train_test_split(classifier_train_raw, test_size=0.2, random_state=random_seed, shuffle=True)
    
    
    split_dataset = [autoencoder_train, autoencoder_val, autoencoder_test, autoencoder_train_raw, autoencoder_val_raw, autoencoder_test_raw,
                     classifier_train, classifier_val, classifier_test, classifier_train_raw, classifier_val_raw, classifier_test_raw]
    return split_dataset

def load_and_split_dataset(eeg_ft_dir = 'ALIGNED_DATA', split_size=100, cutoff_lf=4, random_seed=128, num_subjects = 5, subject_choice_seed=128):
    subject_data_files = glob.glob(os.path.join(eeg_ft_dir, '*.csv'))
    print(eeg_ft_dir)
    # sort the files by the index given to them
    file_name_2_index = lambda file : int(file.split('.')[0].split('_')[-1])
    subject_data_files.sort() # sort alphabetically
    subject_data_files.sort(key=file_name_2_index) # sort by index
    # group data, pick num_subjects randomly
    print(f"Data choices: {subject_data_files}")
    np.random.seed(subject_choice_seed)
    all_eeg_ft_names = np.random.choice(subject_data_files, size=num_subjects, replace=False)
    print(f"Chosen subjects: {all_eeg_ft_names}")
    
    if cutoff_lf is not None:
        sos = signal.butter(10, cutoff_lf, 'lp', fs=1000, output='sos') # low pass filter
    
    
    full_dataset = []
    for x in all_eeg_ft_names:        
        dataset = []
        unfiltered = []
        
        input_label_pair = pd.read_csv(x).values
        input_label_pair_unfiltered = input_label_pair.copy()
        
        if cutoff_lf is not None:
            input_label_pair[:,2:] = signal.sosfiltfilt(sos, input_label_pair[:,2:], axis=0)
        
        n_chunks = len(input_label_pair)/split_size
        
        split_data = np.array_split(input_label_pair, n_chunks).copy()
        split_data_unfiltered = np.array_split(input_label_pair_unfiltered, n_chunks).copy()
        
        del input_label_pair_unfiltered
        del input_label_pair
        
        for x_split, y_split in zip(split_data, split_data_unfiltered):
                if len(x_split) >= split_size:
                    dataset.append(x_split[:split_size].copy())
                    unfiltered.append(y_split[:split_size].copy())
                    
        split_dataset = split_filtered_and_raw(dataset, unfiltered, random_seed) # split data into train, validation and test sets
        
        full_dataset.append(split_dataset)
    return convert_loaded_dataset(full_dataset)

def convert_loaded_dataset(full_dataset):
    '''
    convert the full list into individual numpy arrays
    '''
    return [np.vstack([x[i] for x in full_dataset]) for i in range(12) ]

#################################
#################################


#################################
#################################
#### Models
class autoencoder(nn.Module):
    def __init__(self, num_features=12):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(32, num_features))
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x


def encode_classifier_data(encoder_path, classifier_data_in, encoding):
    ae_model = autoencoder(encoding)
    ae_model.load_state_dict(torch.load(encoder_path))
    ae_model.eval()
    with torch.no_grad():
        prev_shape = classifier_data_in.shape
        x = torch.from_numpy(classifier_data_in[:,:,2:]).float().reshape(-1,1,64) # eeg channels
        x_encoded = ae_model.encode(x).reshape(prev_shape[0], prev_shape[1], encoding).numpy()

        tmp_arr = np.zeros((prev_shape[0], prev_shape[1], 2 + encoding))
        tmp_arr[:, :, :2] = classifier_data_in[:,:,:2].copy() # copy timestamps and feel trace
        tmp_arr[:,:, 2:] = x_encoded # copy encoded eeg
        return tmp_arr



class lstm_classifier(nn.Module):
    def __init__(self, num_features=12, num_hidden=32, dropout=0.2, n_labels=5):
        super(lstm_classifier, self).__init__()
        
        self.hidden_size = num_hidden*2
        self.num_features = num_features
        self.input_size = num_hidden
        self.n_classses = n_labels
        
        self.lstm_1 = nn.LSTM(
            input_size =  self.num_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first=True
        )
        
        self.lstm_2 = nn.LSTM(
            input_size =  self.hidden_size,
            hidden_size = self.input_size,
            num_layers = 1,
            batch_first=True
        )
        
        self.classify = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, self.n_classses))

    
    def forward(self,x):
        x, (h_t, c_t) = self.lstm_1(x)
        x, (h_t, c_t) = self.lstm_2(x)
        x = self.classify(h_t[-1]) # classify last hidden timestep
        return x

#################################
#################################

#################################
#################################
#### Dataloader
def stress_2_label(mean_stress, n_labels=5):
    # value is in [0,1] so map to [0,labels-1] and discretize
    return np.digitize(mean_stress * n_labels, np.arange(n_labels)) - 1

class classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, X, n_labels=5):
        'Initialization'
        self.x = X
        self.n_labels = n_labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index,:,2:]).float() # eeg channels
        y = np.array(stress_2_label(self.x[index, :, 1].mean(axis=-1), n_labels=self.n_labels)).astype(int)
        y = torch.from_numpy(y) # feel trace labels int value [0,n_labels]
        return x, y

#################################
#################################