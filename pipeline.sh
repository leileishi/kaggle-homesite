#!/bin/bash

./build_features.py data/train.csv data/test.csv data/train_feature.csv data/test_feature.csv
./train_lr_model.py data/train_feature.csv data/test_feature.csv models/train_lr.csv models/test_lr.csv --prob --cores 1
./train_rf_model.py data/train_feature.csv data/test_feature.csv models/train_rf.csv models/test_rf.csv --prob --cores 1
./ensemble_models.py data/train.csv ensemble/train_en_feature.csv ensemble/test_en_feature.csv -m models/train_rf.csv models/test_rf.csv models/train_lr.csv models/test_lr.csv
./train_rf_model.py ensemble/train_en_feature.csv ensemble/test_en_feature.csv ensemble/train_en_pred.csv ensemble/test_en_pred.csv --cores 1
./prepare_submit.py data/test.csv ensemble/test_en_pred.csv ensemble/test_submit.csv
