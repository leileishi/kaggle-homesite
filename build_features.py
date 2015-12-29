#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(usage='Build features from training data and test data')

    parser.add_argument("train_input", help="Input file of training data")
    parser.add_argument("test_input", help="Input file of test data")
    parser.add_argument("train_feature", help="Output file of training features")
    parser.add_argument("test_feature", help="Output file of test features")

    args = parser.parse_args()

    df_train_data = pd.read_csv(args.train_input)
    df_test_data = pd.read_csv(args.test_input)

    id_col = df_train_data.columns[0]
    date_col = df_train_data.columns[1]
    target_col = df_train_data.columns[2]
    df_features = df_train_data[df_train_data.columns[3:]]
    col_by_types = df_features.columns.to_series().groupby(df_features.dtypes).groups
    null_cols = df_features.columns[df_features.isnull().sum(axis=0) > 0]
    not_null_cols = df_features.columns[df_features.isnull().sum(axis=0) == 0]
    cat_cols = col_by_types[np.dtype('O')]
    int_cols = col_by_types[np.dtype('int64')]
    float_cols = col_by_types[np.dtype('float64')]

    feature_cols = filter(lambda c: c in not_null_cols, list(int_cols) + list(float_cols))
    feature_target_cols = feature_cols + [target_col]

    df_train_data[feature_target_cols].to_csv(args.train_feature, index=False)
    df_test_data[feature_cols].to_csv(args.test_feature, index=False)

if __name__ == '__main__':
    main()