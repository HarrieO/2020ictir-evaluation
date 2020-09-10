# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import pylab as plt
import json

import models.linear as lnm
import utils.dataset as dataset
import utils.clicks as clk
import utils.ranking as rnk
import utils.pretrained_models as prtr

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str,
                    help="Path to model file.")
parser.add_argument("output_path", type=str,
                    help="Path to output CTR file.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    default=10,
                    help="Length of displayed rankings.")
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='alpha0.1beta0.225')

args = parser.parse_args()

click_model = args.click_model
cutoff = args.cutoff

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                )

fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

pretrain_models = prtr.read_many_models(args.model_file, data)

n_models = pretrain_models.shape[0]

print('Read %d models.' % n_models)


(_,
 train_inv_rankings) = rnk.many_models_data_split_rank_and_invert(
                            pretrain_models,
                            data.train
                          )

print('Finished ranking.')

rel_prob_f = clk.get_relevance_click_model(click_model)
obs_prob = clk.inverse_rank_prob(
                    np.arange(data.train.max_query_size(),
                              dtype=np.float64),
                    1.
                  )
if cutoff:
  obs_prob[cutoff:] = 0.

train_rel_prob = rel_prob_f(data.train.label_vector)
ranker_obs_prob = obs_prob[train_inv_rankings]

CTR_per_ranker = np.sum(train_rel_prob[None, :]*ranker_obs_prob,
                        axis=1, dtype=np.float64)/float(data.train.num_queries())

print(CTR_per_ranker)

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(list(CTR_per_ranker), f)

# fig = plt.figure(figsize=(10, 6), linewidth=0.1)
# plt.hist(CTR_per_ranker, bins=30)
# plt.show()

# fig = plt.figure(figsize=(10, 6), linewidth=0.1)
# plt.hist([CTR_per_ranker[i] - CTR_per_ranker[-1-i] for i in range(int(n_models/2))], bins=30)
# plt.show()