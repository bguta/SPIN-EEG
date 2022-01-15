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

from eeg_utils import *


def train(model, num_epochs, batch_size, learning_rate, train_split, test_split, n_labels=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    train_dataset = classifier_dataset(train_split, n_labels=n_labels)
    test_dataset = classifier_dataset(test_split, n_labels=n_labels)
    
    # figure out class distribution to over sample less represented classes
    train_labels = stress_2_label(train_split[:,:,1].mean(axis=-1), n_labels).astype(int)
    test_labels = stress_2_label(test_split[:,:,1].mean(axis=-1), n_labels).astype(int)
    
    # get the weights of each class as 1/occurance
    train_class_weight= 1/np.bincount(train_labels, minlength=n_labels)
    test_class_weight = 1/np.bincount(test_labels, minlength=n_labels)
    
    # get the per sample weight, which is the likelihood os sampling
    train_sample_weights = [train_class_weight[x] for x in train_labels]
    test_sample_weights = [test_class_weight[x] for x in test_labels]
    
    # sampler
    train_sampler = torch.utils.data.WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)
    test_sampler = torch.utils.data.WeightedRandomSampler(test_sample_weights,  len(test_sample_weights), replacement=True)
    
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               num_workers=8,
                                               sampler=train_sampler)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=batch_size,
                                               num_workers=8,
                                               sampler=test_sampler)
    
    train_metrics = []
    test_metrics = []
    for epoch in range(num_epochs):
        
        # reset metrics
        cur_train_acc = 0 # accuracy
        cur_train_pc = 0 # precision
        cur_train_rc = 0 # recall
        cur_train_f1 = 0 # f1
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
            
            y_hat_np = F.softmax(y_hat.detach(), dim=1).argmax(axis=1).cpu().numpy().squeeze().reshape(-1,) # predictions
            y_np = y.detach().cpu().numpy().squeeze().reshape(-1,) # labels
            
            # metrics
            prf = precision_recall_fscore_support(y_np, y_hat_np, average='macro', zero_division=0)
            
            cur_train_acc += np.mean(y_hat_np == y_np)
            cur_train_pc += prf[0]
            cur_train_rc += prf[1]
            cur_train_f1 += prf[2]
            cur_train_loss += loss.detach().cpu()
        
        # average metrics over loop
        train_loop_size = len(train_loader)
        cur_train_acc  = cur_train_acc/train_loop_size
        cur_train_pc   = cur_train_pc/train_loop_size
        cur_train_rc   = cur_train_rc/train_loop_size
        cur_train_f1   = cur_train_f1/train_loop_size
        cur_train_loss = cur_train_loss/train_loop_size
        
        
        train_metrics.append([cur_train_acc, cur_train_pc, cur_train_rc, cur_train_f1, cur_train_loss])
        
        with torch.no_grad():
            
            # reset metrics
            cur_test_acc = 0 # accuracy
            cur_test_pc = 0 # precision
            cur_test_rc = 0 # recall
            cur_test_f1 = 0 # f1
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
                
                y_hat_np = F.softmax(y_hat.detach(), dim=1).argmax(axis=1).cpu().numpy().squeeze().reshape(-1,)
                y_np = y.detach().cpu().numpy().squeeze().reshape(-1,)
                
                # metrics
                prf = precision_recall_fscore_support(y_np, y_hat_np, average='macro', zero_division=0)
                
                cur_test_acc += np.mean(y_hat_np == y_np)
                cur_test_pc += prf[0]
                cur_test_rc += prf[1]
                cur_test_f1 += prf[2]
                cur_test_loss += loss.detach().cpu()
                
                
            # average metrics over loop
            test_loop_size = len(test_loader)
            cur_test_acc  = cur_test_acc/test_loop_size
            cur_test_pc   = cur_test_pc/test_loop_size
            cur_test_rc   = cur_test_rc/test_loop_size
            cur_test_f1   = cur_test_f1/test_loop_size
            cur_test_loss = cur_test_loss/test_loop_size
        
            test_metrics.append([cur_test_acc, cur_test_pc, cur_test_rc, cur_test_f1, cur_test_loss])
            
        print(f'Epoch:{epoch+1},'\
              f'\nTrain Loss:{cur_train_loss},'\
              f'\nTrain Accuracy:{cur_train_acc},'\
              f'\nTrain Recall: {cur_train_rc},'\
              f'\nTrain precision: {cur_train_pc},' \
              f'\nTrain F1-Score:{cur_train_f1},' \
              f'\nTest Loss:{cur_test_loss},' \
              f'\nTest Accuracy:{cur_test_acc},' \
              f'\nTest Recall: {cur_test_rc},' \
              f'\nTest precision: {cur_test_pc},' \
              f'\nTest F1-Score:{cur_test_f1}')
        
    return train_metrics, test_metrics



import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Train a classifier model to classify emotion state from EEG data using the raw features from the 64 channel EEG')

    argparser.add_argument('--eeg_dir', type=str, default='../ALIGNED_DATA', help='directory containing the EEG and Stress labels post data alignement, example /path/to/data, default value is ../ALIGNED_DATA')
    argparser.add_argument('--time_window', type=int, default=100, help='Integer non-overlapping time window in milliseconds default value is 100')
    argparser.add_argument('--data_split_seed', type=int, default=-1, help='Integer Seed for choosing the datasplit, if -1 uses a random number')
    argparser.add_argument('--num_subjects',  type=int, default=1, help='Number of subjects to include in the dataset, minimum and default is 1 and maximum is 15. Loading large number of users may fail due to memory limitations')
    argparser.add_argument('--subject_seed',  type=int, default=-1, help='Integer seed for choosing the subjects, if -1 uses a random number')
    argparser.add_argument('--classifier_hidden_size',  type=int, default=32, help='Integer LSTM parameter default value is 32, the larger the more complicated the model')
    argparser.add_argument('--classifier_classes',  type=int, default=3, help='Integer number of ouputs for the classifier default value is 3, the interval [0,1] is split into the number of chosen classes')
    argparser.add_argument('--classifier_dropout',  type=float, default=0.0, help='Dropout probability, useful for regularization [0,1] default 0.0')


    argparser.add_argument('--training_epochs',  type=int, default=50, help='Integer epochs for training default 50')
    argparser.add_argument('--training_batch_size',  type=int, default=512, help='Integer batch size for each input during default 512')
    argparser.add_argument('--training_lr',  type=float, default=1e-3, help='Inital learning rate for the model, default 0.001')

    argparser.add_argument('--training_resume_path', type=str, default=None, help='Path to the pretrained model to resume training, example /path/to/cnn_ae_model.pth')
    argparser.add_argument('--training_save_path', type=str, default=None, help='Path to save the trained model to, example /path/to/cnn_ae_model.pth')

    args = argparser.parse_args()

    ### run

    # generate a random seed if -1 is given
    random_seed = rand = int.from_bytes(os.urandom(4), sys.byteorder) if args.data_split_seed == -1 else args.data_split_seed
    subject_choice_seed = int.from_bytes(os.urandom(4), sys.byteorder) if args.subject_seed == -1 else args.subject_seed

    print(f'data_split_seed: {random_seed}')
    print(f'subject_seed: {subject_choice_seed}')

    num_subjects = args.num_subjects
    if num_subjects < 1 or num_subjects > 15:
        print(f"Number chosen subjects {num_subjects} is out of bounds, using 1")
        num_subjects = 1

    # load dataset with no low pass filter applied
    dataset = load_and_split_dataset(eeg_ft_dir = args.eeg_dir, split_size=args.time_window, cutoff_lf=None, random_seed=random_seed, num_subjects = num_subjects, subject_choice_seed=subject_choice_seed)

    classifier_train, classifier_val, classifier_test = dataset[6:9]
    classifier_train_raw, classifier_val_raw, classifier_test_raw = dataset[9:]
    print("Loaded dataset!")

    while True: 
        query = input('ready to train? (y/n)') 
        response = query[0].lower() 
        if query == '' or not response in ['y','n']: 
            print('Please answer with y or n') 
        else: 
            break
    print("Starting to train!")

    '''
    encoder_path = '1_subjects_seed_55_encoder_8_feat.pth'

    classifier_test = encode_classifier_data(encoder_path=encoder_path, classifier_data_in=classifier_test, encoding=8)
    classifier_val = encode_classifier_data(encoder_path=encoder_path, classifier_data_in=classifier_val, encoding=8)
    classifier_train = encode_classifier_data(encoder_path=encoder_path, classifier_data_in=classifier_train, encoding=8)
    
    '''

    time_window=args.time_window # 100ms, non - overlapping
    num_features=64 # 64 channels for input features, eeg input
    num_hidden = args.classifier_hidden_size # LSTM parameter, the larger the more complicated the model
    n_classes = args.classifier_classes # number of ouputs for our classifier, split the range [0,1] into n_classes

    timestamp_start = time.time()
    classifier_model = lstm_classifier(num_features=num_features, num_hidden=num_hidden, dropout=0.0, n_labels=n_classes)

    if args.training_resume_path != '':
        classifier_model.load_state_dict(torch.load(args.training_resume_path))

    model_save_path = args.training_save_path if args.training_save_path != '' else str(int(time.time()*1000.0))+'_raw_classifier_model.pth'

    n_epochs=args.training_epochs
    training_lr = 1e-3 if args.training_lr <= 0.0 else args.training_lr

    train_metrics, test_metrics = train(classifier_model, n_epochs, batch_size=args.training_batch_size, learning_rate=training_lr, train_split=classifier_train, test_split=classifier_val, n_labels=n_classes)


    total_time = time.time() - timestamp_start
    print("Total time(s): {total_time}")
    torch.save(classifier_model.state_dict(), model_save_path)
    print("Saved trained model to: {model_save_path}")


    # Final evaluate
    with torch.no_grad():
        classifier_model.eval()
        x_encoded  = torch.from_numpy(classifier_test[:,:,2:]).float().cuda() # eeg channels
        y = stress_2_label(classifier_test[:,:,1].mean(axis=-1), n_classes).astype(int)
        y_hat = F.softmax(classifier_model(x_encoded).detach(), dim=-1).cpu().numpy()
        labels = y
        preds = y_hat


    prf = precision_recall_fscore_support(labels, np.array([x.argmax() for x in preds]), average='macro', zero_division=0)
    print(f"Precision: {prf[0]}")
    print(f"Recall: {prf[1]}")
    print(f"F1-Score: {prf[2]}")


    # Output confusion matrix
    # fig, axs = plt.subplots(figsize=(20,20), dpi=120)
    # axs.set_title("LSTM Feel Trace Model Confusion Matrix", fontsize=20)
    # axs.set_xlabel("Predicted Label", fontsize=15)
    # axs.set_ylabel("True Label", fontsize=15)

    # cm = confusion_matrix(labels, [x.argmax() for x in preds], labels=np.arange(n_classes), normalize='true')
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(n_classes))

    # disp.plot(ax=axs)
    # plt.show()
    