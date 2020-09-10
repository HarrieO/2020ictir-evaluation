# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import json

import utils.dataset as dataset
import utils.clicks as clk
import utils.ranking as rnk
import utils.pretrained_models as prtr

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str,
                    help="Path to model file.")
parser.add_argument("output_path", type=str,
                    help="Path to output file.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='alpha0.1beta0.225')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--eta", type=float,
                    default=1.0,
                    help="Eta parameter for observance probabilities.")
parser.add_argument("--cutoff", type=int,
                    default=10,
                    help="Length of displayed rankings.")
parser.add_argument("--ranker_pair", type=int,
                    default=0,
                    help="Ranker pair to put in comparison.")

args = parser.parse_args()

click_model = args.click_model
binarize_labels = 'binarized' in click_model
eta = args.eta
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
# chosen_models = np.random.choice(n_models, size=2, replace=False)
chosen_models = np.array([(args.ranker_pair-1)*2, (args.ranker_pair-1)*2+1])

pretrain_models = pretrain_models[chosen_models, :]
n_models = pretrain_models.shape[0]

(test_rankings,
 test_inv_rankings) = rnk.many_models_data_split_rank_and_invert(
                            pretrain_models,
                            data.test
                          )

rel_prob_f = clk.get_relevance_click_model(click_model)
obs_prob = clk.inverse_rank_prob(
                    np.arange(max(
                        data.train.max_query_size(),
                        data.validation.max_query_size(),
                        data.test.max_query_size(),
                      ), dtype=np.float64),
                    eta
                  )
if cutoff:
  obs_prob[cutoff:] = 0.

test_rel_prob = rel_prob_f(data.test.label_vector)
ranker_test_ctr = np.zeros(n_models, dtype=np.float64)
for i in range(n_models):
  ranker_test_ctr[i] = np.mean(obs_prob[test_inv_rankings[i]]*test_rel_prob)
  ranker_test_ctr[i] *= data.test.num_docs()/float(data.test.num_queries())

(models_train_rankings,
 models_train_inv_rankings) = rnk.many_models_data_split_rank_and_invert(
                                        pretrain_models,
                                        data.train,
                                      )
# model_train_doc_values = 1./np.log2(models_train_inv_rankings.astype(np.float64) + 2.)
model_train_doc_values = obs_prob[models_train_inv_rankings]
train_doc_value_diff = model_train_doc_values[0, :] - model_train_doc_values[1, :]

(models_vali_rankings,
 models_vali_inv_rankings) = rnk.many_models_data_split_rank_and_invert(
                                    pretrain_models,
                                    data.validation,
                                  )
# model_vali_doc_values = 1./np.log2(models_vali_inv_rankings.astype(np.float64) + 2.)
model_vali_doc_values = obs_prob[models_vali_inv_rankings]
vali_doc_value_diff = model_vali_doc_values[0, :] - model_vali_doc_values[1, :]

rel_train_prob = rel_prob_f(data.train.label_vector)
rel_vali_prob = rel_prob_f(data.validation.label_vector)

obs_rel_mul = vali_doc_value_diff*rel_vali_prob
expected_vali_reward = np.mean(obs_rel_mul, dtype=np.float64)
expected_vali_reward *= data.validation.num_docs()/float(data.validation.num_queries())

print('Expected Validation Mean')
print(expected_vali_reward)

obs_rel_mul = train_doc_value_diff*rel_train_prob
expected_train_reward = np.mean(obs_rel_mul, dtype=np.float64)
expected_train_reward *= data.train.num_docs()/float(data.train.num_queries())


print('Expected Train Mean')
print(expected_train_reward)

test_diff = ranker_test_ctr[0] - ranker_test_ctr[1]
print('Test Mean: %0.04f' % test_diff)

doc_per_q = data.train.query_sizes()
cutoff_n_min = np.minimum(doc_per_q, cutoff+1)
n_doc_pos = np.sum(doc_per_q*cutoff_n_min)

results = {
  'model name': 'A/B',
  'model type': 'A/B',
  'click model': click_model,
  'dataset name': args.dataset,
  'fold': fold_id,
  'train difference': expected_train_reward,
  'validation difference': expected_vali_reward,
  'test difference': test_diff,
  'results':[],
  }

n_queries_to_sample = 3*10**6
measure_points = np.unique(np.array(
                sorted(list(np.unique(
                  np.geomspace(10**2, n_queries_to_sample, 500).astype(np.int64))) 
                      + [10**x for x in range(2,7)])))

n_models = models_train_rankings.shape[0]
clicks_per_model = np.zeros(2, dtype=np.int64)
samples_per_model = np.zeros(2, dtype=np.int64)
estimates = np.zeros(n_queries_to_sample, dtype=np.float64)
for sample_i in range(n_queries_to_sample):
  qid = np.random.choice(data.train.num_queries())
  q_n_docs = data.train.query_size(qid)
  ranking_len = np.minimum(q_n_docs, cutoff+1)

  model_i = np.random.choice(n_models)
  s_i, e_i = data.train.query_range(qid)
  inv_ranking = models_train_inv_rankings[model_i, s_i:e_i]

  click_prob = obs_prob[inv_ranking]*rel_train_prob[s_i:e_i]
  clicks = clk.bernoilli_sample_from_probs(click_prob)

  samples_per_model[model_i] += 1
  clicks_per_model[model_i] += np.sum(clicks)

  if np.any(clicks):
    if model_i == 0:
      estimates[sample_i] += 2*np.sum(clicks)
    else:
      estimates[sample_i] -= 2*np.sum(clicks)

  # if (sample_i+1) % 1000 == 0:
  if (sample_i+1) == measure_points[0]:
    measure_points = measure_points[1:]

    model_estimates = clicks_per_model.astype(np.float64)/np.maximum(1.,samples_per_model.astype(np.float64))
    estimate = model_estimates[0] - model_estimates[1]
    bin_estimate = np.sign(model_estimates[0] - model_estimates[1])

    # print('Estimate from %d clicks, estimate: %0.05f squared error: %0.09f error: %0.09f' % (
    #         sample_i+1, estimate,
    #         (estimate - expected_train_reward)**2.,
    #         (estimate - expected_train_reward),
    #         ))

    # print('Deviation from true mean in estimate: %0.09f' % np.mean((estimates[:sample_i] - expected_train_reward)**2.) )
    # print('Old squared error: %0.09f' % (estimate_1 - expected_train_reward)**2.)
    # print('Mean error from true mean in estimate: %0.05f' % np.mean((estimates[:sample_i] - expected_train_reward)) )
    # print('Deviation from validation mean in estimate: %0.05f' % np.mean((estimates[:click_i] - expected_vali_reward)**2.) )

    results['results'].append(
      {
        'num queries': sample_i+1,
        'binary error': float(bin_estimate != np.sign(test_diff)),
        'binary train error': float(bin_estimate != np.sign(expected_train_reward)),
        'squared error': (estimate - expected_train_reward)**2.,
        'absolute error': np.abs(estimate - expected_train_reward),
        'estimate': estimate,
        'mean squared error': np.mean((estimates[:sample_i] - expected_train_reward)**2.),
        'logging policy CTR': np.sum(clicks_per_model)/np.float64(sample_i+1),
      }
    )

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(results, f)
