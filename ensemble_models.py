#!/usr/bin/env python

import sys
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(usage='Build ensemble features from existing model outputs')

    parser.add_argument("train_input", help="Input file of training data")
    parser.add_argument("train_en_feature", help="Output file of training ensemble features and target")
    parser.add_argument("test_en_feature", help="Output file of test ensemble features")
    parser.add_argument('-m','--models', nargs='+', help='Model files in pairs of (train_pred, test_pred)')

    args = parser.parse_args()

    df_train_data = pd.read_csv(args.train_input)
    target_col = df_train_data.columns[2]
    df_target = df_train_data[target_col]

    train_en_features = map(lambda f: pd.read_csv(f), args.models[0::2])
    test_en_features = map(lambda f: pd.read_csv(f), args.models[1::2])
    df_train_en_feature_target = pd.concat(train_en_features + [df_target], axis=1)
    df_test_en_feature = pd.concat(test_en_features, axis=1)

    feature_names = map(lambda f: f.split('.')[0].replace('/', '_'), args.models[0::2])
    df_train_en_feature_target.columns = feature_names + [target_col]
    df_test_en_feature.columns = feature_names

    df_train_en_feature_target.to_csv(args.train_en_feature, index=False)
    df_test_en_feature.to_csv(args.test_en_feature, index=False)

if __name__ == '__main__':
    main()