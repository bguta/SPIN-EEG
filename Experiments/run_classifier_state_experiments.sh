#!/bin/bash
for i in {1..5}
do
	python3 classifier_emotion_state.py --classifier_dropout 0.1 --classifier_classes 5 --num_subjects 10 --time_window 100 --skip_prompt 1
done
