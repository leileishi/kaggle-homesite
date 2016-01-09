#!/bin/bash

date > log.txt

./build_features.py data/train.csv data/test.csv data/train_feature.csv data/test_feature.csv
unbuffer ./train_rf_model.py data/train_feature.csv data/test_feature.csv models/rf.csv --prob --cores 1 | tee -a log.txt
unbuffer ./train_ada_model.py data/train_feature.csv data/test_feature.csv models/ada.csv --prob --cores 1 | tee -a log.txt
unbuffer ./train_gbc_model.py data/train_feature.csv data/test_feature.csv models/gbc.csv --prob --cores 1 | tee -a log.txt
unbuffer ./train_xgb_model.py data/train_feature.csv data/test_feature.csv models/xgb.csv --prob --cores 1 | tee -a log.txt

rm models/models.csv
echo "Model,Weight" > models/models.csv
cat log.txt | grep "~~" | sed 's/~~/,/g' >> models/models.csv

./average_rank.py models/models.csv ensemble/test_en_pred.csv
./prepare_submit.py data/test.csv ensemble/test_en_pred.csv ensemble/test_submit.csv