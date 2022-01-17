import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm
import time
import argparse
import json
from eeg_utils import *


def train(model, num_epochs, batch_size, learning_rate, train_split, test_split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)

    criterion = nn.MSELoss()
    
    train_dataset = autoencoder_dataset(train_split)
    test_dataset = autoencoder_dataset(test_split)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True)
    
    train_metrics = []
    test_metrics = []
    for epoch in range(num_epochs):
        
        # reset metrics
        cur_train_loss = 0 # loss
        
        # set to train mode
        model.train()
        
        # loop over dataset
        for data in tqdm(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # metrics
            cur_train_loss += loss.detach().cpu()
            scheduler.step()
        
        # average metrics over loop
        train_loop_size = len(train_loader)
        cur_train_loss = cur_train_loss/train_loop_size
        
        
        train_metrics.append([cur_train_loss])
        
        with torch.no_grad():
            
            # reset metrics
            cur_test_loss = 0 # loss
            
            # set to evaluate mode, ignores dropout
            model.eval()
            
            # loop over dataset
            for data in tqdm(test_loader):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                
                y_hat = model(x)
                loss = criterion(y_hat, y)
                
                # metrics
                cur_test_loss += loss.detach().cpu()
                
                
            # average metrics over loop
            test_loop_size = len(test_loader)
            cur_test_loss = cur_test_loss/test_loop_size
        
            test_metrics.append([cur_test_loss])
            
        print(f'Epoch:{epoch+1},'\
              f'\nTrain Loss:{cur_train_loss},'\
              f'\nTest Loss:{cur_test_loss},')
        
    return train_metrics, test_metrics

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Train an autoencoder model to encode the EEG signals into a lower dimensional space to be used as input into a classifier')

    argparser.add_argument('--eeg_dir', type=str, default='../ALIGNED_DATA', help='directory containing the EEG and Stress labels post data alignement, example /path/to/data, default value is ../ALIGNED_DATA')
    argparser.add_argument('--time_window', type=int, default=100, help='Integer non-overlapping time window in milliseconds default value is 100')
    argparser.add_argument('--data_split_seed', type=int, default=-1, help='Integer Seed for choosing the datasplit, if -1 uses a random number')
    argparser.add_argument('--num_subjects',  type=int, default=1, help='Number of subjects to include in the dataset, minimum and default is 1 and maximum is 15. Loading large number of users may fail due to memory limitations')
    argparser.add_argument('--subject_seed',  type=int, default=-1, help='Integer seed for choosing the subjects, if -1 uses a random number')
    argparser.add_argument('--eeg_low_pass', type=int, default=None, help='-3 dB point frequency (Hz) for non-causal low pass filter, by default no filter is applied')
    argparser.add_argument('--encoder_feature_size',  type=int, default=8, help='Integer Autoencoder encoding size default value is 8, the larger the more complicated the model')

    argparser.add_argument('--training_epochs',  type=int, default=50, help='Integer epochs for training default 50')
    argparser.add_argument('--training_batch_size',  type=int, default=512, help='Integer batch size for each input during default 512')
    argparser.add_argument('--training_lr',  type=float, default=1e-3, help='Inital learning rate for the model, default 0.001')

    argparser.add_argument('--training_resume_path', type=str, default=None, help='Path to the pretrained model to resume training, example /path/to/cnn_ae_model.pth')
    argparser.add_argument('--training_save_path', type=str, default=None, help='Path to save the trained model to, example /path/to/cnn_ae_model.pth')
    argparser.add_argument('--skip_prompt', type=int, default=0, help='skip ready to train prompt option 0(false) or 1(true), default 0')
    
    args = argparser.parse_args()

    ### run

    # generate a random seed if -1 is given
    random_seed = rand = int.from_bytes(os.urandom(4), sys.byteorder) if args.data_split_seed == -1 else args.data_split_seed
    subject_choice_seed = int.from_bytes(os.urandom(4), sys.byteorder) if args.subject_seed == -1 else args.subject_seed

    print(f'data_split_seed: {random_seed}')
    print(f'subject_seed: {subject_choice_seed}')

    num_subjects = args.num_subjects
    if num_subjects < 1 or num_subjects > 16:
        print(f"Number chosen subjects {num_subjects} is out of bounds, using 1")
        num_subjects = 1
    
    cutoff_lf = args.eeg_low_pass
    # load dataset with no low pass filter applied
    dataset = load_and_split_dataset(eeg_ft_dir = args.eeg_dir, split_size=args.time_window, cutoff_lf=cutoff_lf, random_seed=random_seed, num_subjects = num_subjects, subject_choice_seed=subject_choice_seed)

    autoencoder_train, autoencoder_val, autoencoder_test = dataset[:3]
    autoencoder_train_raw, autoencoder_val_raw, autoencoder_test_raw = dataset[3:6]
    print("Loaded dataset!")

    time_window=args.time_window # 100ms, non - overlapping
    num_hidden = args.encoder_feature_size # output encoding size

    timestamp_start = time.time()
    encoder_model = autoencoder(num_features=num_hidden)

    if args.training_resume_path != None and '.pth' in args.training_resume_path:
        encoder_model.load_state_dict(torch.load(args.training_resume_path))

    model_save_path = args.training_save_path if args.training_save_path != None and '.pth' in args.training_save_path  else str(int(time.time()*1000.0))+'_encoder_model.pth'

    n_epochs=args.training_epochs
    training_lr = 1e-3 if args.training_lr <= 0.0 else args.training_lr

    print("###################################")
    print("Parameters:")
    param_dict = {k:v for k,v in vars(args).items()}
    param_dict['data_split_seed'] = random_seed
    param_dict['subject_seed'] = subject_choice_seed

    with open(model_save_path.replace('.pth','.json'), 'w') as fp:
        json.dump(param_dict, fp)
    print(param_dict)

    while True and not args.skip_prompt: 
        query = input('ready to train? (y/n)') 
        response = query[0].lower() 
        if query == '' or not response in ['y']: 
            print('Please answer with y to start training') 
        else: 
            break
    print("###################################")
    print("###################################")
    print("Starting to train!")

    train_metrics, test_metrics = train(encoder_model, n_epochs, batch_size=args.training_batch_size, learning_rate=training_lr, train_split=autoencoder_train, test_split=autoencoder_val)

    total_time = time.time() - timestamp_start
    print("###################################")
    print("###################################")
    print("Finished Training!")
    print(f"Total time(s): {total_time}")
    torch.save(encoder_model.state_dict(), model_save_path)
    print(f"Saved trained model to: {model_save_path}")


    # Final evaluate
    print("###################################")
    print("Metrics:")

    channel_errs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoder_model.eval()
        prev_shape = autoencoder_test.shape
        x = torch.from_numpy(autoencoder_test[:,:,2:]).float().reshape(-1,1,64).to(device) # eeg channels
        x_encoded = encoder_model(x).reshape(prev_shape[0], prev_shape[1], 64).detach().cpu().numpy().squeeze()

        print(f"Test Mean Square Error: {((autoencoder_test[:,:,2:] - x_encoded)**2).mean()}")

        for data_index in range(64):
            t = autoencoder_test[:, :, 0].flatten()
            ind_t = np.argsort(t)
            t = t[ind_t]
            z_raw = autoencoder_test_raw[:, :, data_index+2].flatten()[ind_t]
            z = autoencoder_test[:, :, data_index+2].flatten()[ind_t]
            z_hat = x_encoded[:, :, data_index].flatten()[ind_t]

            max_error = np.max(np.abs(z_hat - z))
            channel_errs.append(max_error)

    # Output Channel Errors matrix
    fig, axs = plt.subplots(figsize=(20,15))
    axs.grid(True)
    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.set_title(f"Absolute Max Error for Each Channel", fontsize=20)
    axs.set_xlabel("Channel #", fontsize=20)
    axs.set_ylabel("Absolute Max Error", fontsize=20)
    axs.bar(np.arange(len(channel_errs)), channel_errs, alpha=0.7, label='Absolute Max Error')

    axs.legend(loc='best', fontsize=20)
    fig.savefig(model_save_path.replace('.pth','.png'))
    #plt.show()