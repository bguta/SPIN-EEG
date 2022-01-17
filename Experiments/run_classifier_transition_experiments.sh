#!/bin/bash
for i in {1..5}
do
	python3 classifier_emotion_transition.py --classifier_dropout 0.1 --classifier_classes 3 --num_subjects 10 --time_window 1000 --skip_prompt 1
done
