import pandas as pd
import numpy as np
import scipy.io as sp_io
import os
import glob
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed



@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def normalize_and_crop(eeg: np.array, ft: np.array) -> tuple:
    """
    :param eeg:
    :param ft:
    :return:
    """
    min_index = np.where(ft[:, 0] >= 0)[0][0]  # first positive time stamp
    max_index = int(min(eeg[:, 0][-2], ft[:, 0][-1]))
    # find the index on ft that is greater than or equal to the max index timestamp
    ft_end_index = np.where(ft[:, 0] > max_index)[0][0] - 1 if ft[:, 0][-1] > eeg[:, 0][-2] else \
        np.where(ft[:, 0] == max_index)[0][0]

    new_ft = ft[min_index:ft_end_index + 1, :].copy()
    new_eeg = eeg[int(ft[:, 0][min_index]):max_index + 1, :-1].copy()

    # normalize to be between [0,1]
    min_eeg = np.min(new_eeg[:, 1:], axis=0)
    max_eeg = np.max(new_eeg[:, 1:], axis=0)

    min_ft = 0
    max_ft = 225

    new_eeg[:, 1:] = (new_eeg[:, 1:] - min_eeg) / (max_eeg - min_eeg)
    new_ft[:, 1] = (new_ft[:, 1] - min_ft) / (max_ft - min_ft)

    new_ft[:, 0] = new_ft[:, 0] / 1000  # ms to seconds
    new_eeg[:, 0] = new_eeg[:, 0] / 1000  # ms to seconds

    return new_eeg, new_ft


def create_dataset(src_dir: str, out_dir = 'EEG_FT_DATA') -> None:
    """
    :param src_dir: the directory containing all the EEG and FeelTrace data in folders for each subject

    Creates the normalized and cropped dataset in the EEG_FT_DATA directory, throws an error if
    EEG_FT_DATA does not exist
    """
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))

    subject_data = [glob.glob(os.path.join(x, '*')) for x in subject_data_dir]

    all_eeg_data = [ next(filter(lambda item: 'eeg.mat' in item and 'eeg_eeg.mat' not in item, x))for x in subject_data] # find all eeg.mat
    all_joystick_data = [ next(filter(lambda item: 'joystick.mat' in item and 'joystick_joystick.mat' not in item, x)) for x in subject_data] # final all joystick.mat

    # the next steps takes a bit of time!!
    eeg_ft_pairs = [(eeg, ft) for eeg, ft in zip(all_eeg_data, all_joystick_data)]

    with tqdm_joblib(tqdm(desc="Dataset Creation", total=len(eeg_ft_pairs))) as progress_bar:
        Parallel(n_jobs=8)(delayed(write_to_csv_dataset_loop)(i, x, y, out_dir) for i, (x, y) in enumerate(eeg_ft_pairs))
    print(f'Created dataset initial dataset csv in {out_dir}')


def write_to_csv_dataset_loop(index: int, x: str, y: str, out_dir) -> None:
    """
    Should not be called by the user, for pair at index, create the pandas dataframe and write to a csv file
    :param y: FeelTrace filename
    :param x: EEG filename
    :param index: index of the pair to write
    """

    eeg_column_headers = ['t'] + [f'channel_{i}' for i in range(64)]
    ft_column_headers = ['t', 'stress']

    eeg, ft = sp_io.loadmat(x)['var'], sp_io.loadmat(y)['var']
    normalized_eeg, normalized_ft = normalize_and_crop(eeg, ft)

    eeg_df = pd.DataFrame(data=normalized_eeg, columns=eeg_column_headers)
    ft_df = pd.DataFrame(data=normalized_ft, columns=ft_column_headers)

    eeg_df.to_csv(os.path.join(out_dir, f'normalized_eeg_{index}.csv'), index=False)
    ft_df.to_csv(os.path.join(out_dir, f'normalized_ft_{index}.csv'), index=False)


def create_filled_lables(ft_df, eeg_df):
    """
    Fill feel trace missing values so that we have a value for every ms.
    We do this by repeating the last value until there is a change.
    """
    ft_arr = np.ones_like(eeg_df[:,:2]) * -1 # fill array with -1 to indicate values to fill
    ft_arr[:,0] = eeg_df[:,0] # fill time values in
    ft_arr_timestamps = ft_df[:,0] * 1000 # convert timestamps to indicies
    index_arr = (ft_arr_timestamps - ft_arr_timestamps[0]).astype(int) # actual convertion here, we remove offset
    
    ft_arr[index_arr,1] =  ft_df[:,1] # fill the known values in
    
    not_missing_mask = ft_arr[:,1] != -1 # non -1 values
    non_missing_index = np.where(not_missing_mask, np.arange(len(ft_arr[:,1])), -1) # obtain index of non missing values
    existing_val = np.maximum.accumulate(non_missing_index) # find max value index which corresponds to the first non -1 value in each interval
    
    ft_arr[not_missing_mask == False, 1] = ft_arr[existing_val[not_missing_mask == False], 1]
    
    return np.hstack((ft_arr, eeg_df[:,1:]))

def save_aligned_dataset(eeg_ft_dir = 'EEG_FT_DATA', output_dir='ALIGNED_DATA'):
    """
    Save the dataset generated by filling in the missing feel trace values to disk.
    We call this new dataset "aligned"
    """
    subject_data_files = glob.glob(os.path.join(eeg_ft_dir, '*.csv'))
    # sort the files by the index given to them
    file_name_2_index = lambda file : int(file.split('.csv')[0].split('_')[-1])
    subject_data_files.sort() # sort alphabetically
    subject_data_files.sort(key=file_name_2_index) # sort by index
    # group the eeg and ft into pairs
    subject_data_pairs = [(subject_data_files[i], subject_data_files[i+1]) for i in range(0, len(subject_data_files) - 1, 2)]
    all_eeg_ft_names = [(x[0],x[1]) for x in subject_data_pairs]
    
    column_headers = ['t', 'stress'] + [f'channel_{i}' for i in range(64)]

    index = 0
    for x in tqdm(all_eeg_ft_names):
        tmp_read_ft = pd.read_csv(x[1]).values
        tmp_read_eeg = pd.read_csv(x[0]).values
        
        input_label_pair = create_filled_lables(tmp_read_ft, tmp_read_eeg)
        
        df = pd.DataFrame(data=input_label_pair, columns=column_headers)
        
        df.to_csv(os.path.join(output_dir, f'EEG_FT_ALIGNED_{index}.csv'), index=False)
        
        del tmp_read_ft
        del tmp_read_eeg
        del input_label_pair
        
        index += 1
    print(f'Done! Created final dataset in {output_dir}')