#!/usr/bin/env python

import sys
import argparse
import numpy as np
from scipy import stats
import pandas as pd

def main():
    parser = argparse.ArgumentParser(usage='Average ranks of model outputs')

    parser.add_argument("model_list", help="Input file of models and weights")
    parser.add_argument("test_pred", help="Output file of predictions")

    args = parser.parse_args()

    print('Average ranks of model outputs')

    model_list = pd.read_csv(args.model_list)
    models = map(lambda m: pd.read_csv(m), model_list.Model)
    ranks = map(lambda m: m.values.T[0].argsort().argsort(), models)
    weighted_rank = map(lambda (r, w): r**(1/w), zip(ranks, model_list.Weight.values))
    average_rank = stats.gmean(weighted_rank, axis=0)
    norm_rank = average_rank/max(average_rank)

    pd.Series(norm_rank, name='Prob').to_csv(args.test_pred, index=False, header=True)

if __name__ == '__main__':
    main()