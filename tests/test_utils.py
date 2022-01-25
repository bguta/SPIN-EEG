import sys
import os
import numpy as np
  
# appending a path
sys.path.append('..')

from Experiments.eeg_utils import stress_2_label_transition, stress_2_label
from src.utils import filter_normalize_and_crop

def test_eeg_dataset_creation():
    """
    Test filter and normalization functionality
    """
    size = 20 * 100
    eeg_data = np.abs(np.random.normal(0, 10, size=(size,66)) * 1e6)
    ft_data = np.random.normal(0, 1, size=(size,2))

    eeg_data[:,0] = np.arange(size)
    ft_data[:,0] = np.arange(size) - 1e3

    normalized_eeg, normalized_ft = filter_normalize_and_crop(eeg_data, ft_data)

    # crop out negative indicies
    assert normalized_eeg.shape[0] == size - 1e3
    assert normalized_ft.shape[0] == size - 1e3


def test_classifier_utils():
    """
    Test class label creation
    """
    stress_avg = np.array([0, 0.19, 0.2, 0.4, 0.6, 0.8, 1.0])
    stress_avg_labels = stress_2_label(stress_avg, n_labels=5)


    stress_slope = np.array([-10, -5, -0.1, 0, 0.1, 5, 10])
    stress_slope_labels = stress_2_label_transition(stress_slope)

    assert np.allclose(np.array([0, 0, 1, 2, 3, 4, 4]),stress_avg_labels)
    assert np.allclose(stress_slope_labels, np.array([0, 0, 0, 1, 2, 2, 2]))
