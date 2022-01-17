#!/bin/bash
for i in {1..5}
do
	python3 autoencoder_eeg.py --num_subjects 10 --time_window 100 --encoder_feature_size 12 --skip_prompt 1
done
