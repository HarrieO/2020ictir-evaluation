# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import json
import os

import models.linear as lnm
import policies.plackettluce as pl
import utils.dataset as dataset
import utils.clicks as clk
import utils.ranking as rnk
import utils.variance as var
import utils.pretrained_models as prtr

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
parser.add_argument("--update_steps", type=int,
                    default=10**4,
                    help="Number of gradient descent steps to take per update.")
parser.add_argument("--ranker_pair", type=int,
                    default=0,
                    help="Ranker pair to put in comparison.")
parser.add_argument("--neural", action='store_true',
                    help="Train a neural policy.")

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

(_, models_train_inv_rankings) = rnk.many_models_data_split_rank_and_invert(
                                        pretrain_models,
                                        data.train,
                                      )
# model_train_doc_values = 1./np.log2(models_train_inv_rankings.astype(np.float64) + 2.)
model_train_doc_values = obs_prob[models_train_inv_rankings]
train_doc_value_diff = model_train_doc_values[0, :] - model_train_doc_values[1, :]

(_, models_vali_inv_rankings) = rnk.many_models_data_split_rank_and_invert(
                                    pretrain_models,
                                    data.validation,
                                  )
# model_vali_doc_values = 1./np.log2(models_vali_inv_rankings.astype(np.float64) + 2.)
model_vali_doc_values = obs_prob[models_vali_inv_rankings]
vali_doc_value_diff = model_vali_doc_values[0, :] - model_vali_doc_values[1, :]

scale_factor = np.amax(np.abs(train_doc_value_diff))
# train_doc_value_diff /= scale_factor/100.
# vali_doc_value_diff /= scale_factor/100.

additional_train_feat = np.stack((
                            train_doc_value_diff,
                            np.abs(train_doc_value_diff),
                            train_doc_value_diff**2.,
                            np.equal(train_doc_value_diff, 0),
                            np.greater(train_doc_value_diff, 0),
                            models_train_inv_rankings[0, :],
                            models_train_inv_rankings[1, :],
                            np.zeros(data.train.num_docs())
                            ), axis=1,)
for qid in range(data.train.num_queries()):
  s_i, e_i = data.train.query_range(qid)
  additional_train_feat[s_i:e_i, :] -= np.amin(additional_train_feat[s_i:e_i, :], axis=0)[None, :]
  max_denom = np.amax(additional_train_feat[s_i:e_i, :], axis=0)
  max_denom[np.equal(max_denom, 0.)] = 1.
  additional_train_feat[s_i:e_i, :] /= max_denom[None, :]
  additional_train_feat[s_i:e_i, -1] = (s_i - e_i)
additional_vali_feat = np.stack((
                            vali_doc_value_diff,
                            np.abs(vali_doc_value_diff),
                            vali_doc_value_diff**2.,
                            np.equal(vali_doc_value_diff, 0),
                            np.greater(vali_doc_value_diff, 0),
                            models_vali_inv_rankings[0, :],
                            models_vali_inv_rankings[1, :],
                            np.zeros(data.validation.num_docs()),
                            ), axis=1,)
for qid in range(data.validation.num_queries()):
  s_i, e_i = data.validation.query_range(qid)
  additional_vali_feat[s_i:e_i, :] -= np.amin(additional_vali_feat[s_i:e_i, :], axis=0)[None, :]
  max_denom = np.amax(additional_vali_feat[s_i:e_i, :], axis=0)
  max_denom[np.equal(max_denom, 0.)] = 1.
  additional_vali_feat[s_i:e_i, :] /= max_denom[None, :]
  additional_vali_feat[s_i:e_i, -1] = (s_i - e_i)

n_feat = data.num_features + additional_train_feat.shape[1]
if args.neural:
  import models.neural as dnnm
  logging_policy = dnnm.NeuralModel(n_feat)
else:
  logging_policy = lnm.LinearModel(n_feat)

logging_vali_scores = logging_policy.score_data_split(
                          data.validation,
                          additional_feat=additional_vali_feat,
                        )

sample_vali_f = lambda x: pl.sample_rankings_query(
                                data.validation,
                                x,
                                logging_vali_scores,
                                10**3,
                                cutoff=cutoff,
                                prob_per_rank=True,
                              )
rel_train_prob = rel_prob_f(data.train.label_vector)
rel_vali_prob = rel_prob_f(data.validation.label_vector)

obs_rel_mul = vali_doc_value_diff*rel_vali_prob
expected_vali_reward = np.mean(obs_rel_mul, dtype=np.float64)
expected_vali_reward *= data.validation.num_docs()/float(data.validation.num_queries())

print('Expected Validation Mean')
print(expected_vali_reward)

vali_variance = var.oracle_data_split_list_variance(
                      data.validation,
                      sample_vali_f,
                      expected_vali_reward,
                      vali_doc_value_diff,
                      rel_vali_prob,
                      obs_prob,
                      logging_vali_scores,
                      cutoff=cutoff,
                    )

obs_rel_mul = train_doc_value_diff*rel_train_prob
expected_train_reward = np.mean(obs_rel_mul, dtype=np.float64)
expected_train_reward *= data.train.num_docs()/float(data.train.num_queries())


print('Expected Train Mean')
print(expected_train_reward)

test_diff = ranker_test_ctr[0] - ranker_test_ctr[1]
print('Test Mean: %0.04f' % test_diff)

start_variance = vali_variance
print('Updates 0: %s' % vali_variance)

update_i = 0
for update_i in range(args.update_steps):
  qid = np.random.choice(data.train.num_queries())
  q_n_docs = data.train.query_size(qid)
  s_i, e_i = data.train.query_range(qid)
  q_value_diff = train_doc_value_diff[s_i:e_i]

  update_i += 1
  if np.all(np.equal(q_value_diff, 0.)):
    continue

  policy_query_scores = logging_policy.score_query(
                            qid,
                            data.train,
                            additional_feat=additional_train_feat,
                          )
  policy_query_scores += 18 - np.amax(policy_query_scores)

  (sampled_rankings,
   sampled_inv_rankings,
   sampled_ranking_prob,
   prob_per_rank) = pl.sample_rankings(
                                policy_query_scores,
                                10**3,
                                cutoff=cutoff,
                                prob_per_rank=True,
                              )

  doc_prop_scores = np.sum(prob_per_rank*obs_prob[:prob_per_rank.shape[0], None], axis=0)

  (list_variance,
   list_var_score_grad,
   list_var_policy_grad) = var.oracle_list_variance(
                                        expected_train_reward,
                                        train_doc_value_diff[s_i:e_i],
                                        rel_train_prob[s_i:e_i],
                                        obs_prob,
                                        doc_prop_scores,
                                        policy_query_scores,
                                        sampled_rankings,
                                        sampled_inv_rankings,
                                        sampled_ranking_prob,
                                        cutoff=cutoff,
                                        # compute_gradient=False,
                                        )

  policy_grad = pl.gradient_based_on_samples(
                      sampled_rankings,
                      obs_prob,
                      policy_query_scores,
                      sampled_ranking_prob,
                      cutoff=cutoff,
                    )

  score_grad = np.mean(list_var_policy_grad[:, None]*policy_grad, axis=0)
  score_grad += list_var_score_grad

  logging_policy.gradient_update(
                    -score_grad,
                    data.train.query_feat(qid),
                    learning_rate=10**-2,
                    additional_feat=additional_train_feat[s_i:e_i, :],
                  )

logging_vali_scores[:] = logging_policy.score_data_split(
                            data.validation,
                            additional_feat=additional_vali_feat,
                          )

vali_variance = var.oracle_data_split_list_variance(
                      data.validation,
                      sample_vali_f,
                      expected_vali_reward,
                      vali_doc_value_diff,
                      rel_vali_prob,
                      obs_prob,
                      logging_vali_scores,
                      cutoff=cutoff,
                    )
print('Updates %04d: %0.05f (%0.02f speedup ; %0.02f variance-reduction)' % (update_i ,
                                       vali_variance,
                                       start_variance/vali_variance,
                                       vali_variance/start_variance))

policy_scores = logging_policy.score_data_split(
                              data.train,
                              additional_feat=additional_train_feat,
                            )

doc_per_q = data.train.query_sizes()
cutoff_n_min = np.minimum(doc_per_q, cutoff+1)
n_doc_pos = np.sum(doc_per_q*cutoff_n_min)
docpos_ranges = np.zeros(data.train.num_queries()+1, dtype=np.int64)
docpos_ranges[1:] = np.cumsum(doc_per_q*cutoff_n_min)

results = {
  'number of updates': args.update_steps,
  'model name': 'Oracle Linear (%d updates)' % args.update_steps,
  'model type': 'Oracle Linear',
  'click model': click_model,
  'dataset name': args.dataset,
  'fold': fold_id,
  'train difference': expected_train_reward,
  'validation difference': expected_vali_reward,
  'test difference': test_diff,
  'validation variance': vali_variance,
  'validation variance ratio': start_variance/vali_variance,
  'validation variance reduction': vali_variance/start_variance,
  'results': [],
    # [{
    #   'num queries': 0,
    #   'binary error': 0.5,
    #   'binary train error': 0.5,
    #   'squared error': 0.5,
    #   'absolute error': 0.5,
    # }]
  }
if args.neural:
  results['model name'] = 'Oracle Neural (%d updates)' % args.update_steps
  results['model type'] = 'Oracle Neural'
elif args.update_steps == 0:
  results['model name'] = 'Uniform Logging'
  results['model type'] = 'Uniform Logging'



n_queries_to_sample = 3*10**6
measure_points = np.unique(np.array(
                sorted(list(np.unique(
                  np.geomspace(10**2, n_queries_to_sample, 500).astype(np.int64))) 
                      + [10**x for x in range(2,7)])))

total_clicks = 0
estimates = np.zeros(n_queries_to_sample, dtype=np.float64)
prop_computed = np.zeros(data.train.num_queries(), dtype=bool)
all_prop_scores = np.zeros(data.train.num_docs(), dtype=np.float64)
for sample_i in range(n_queries_to_sample):
  qid = np.random.choice(data.train.num_queries())
  q_n_docs = data.train.query_size(qid)
  ranking_len = np.minimum(q_n_docs, cutoff+1)

  inv_ranking = pl.sample_rankings_query(
                      data.train,
                      qid,
                      policy_scores,
                      1,
                      cutoff=cutoff,
                    )[1][0,:]
  
  # rel_prob 
  s_i, e_i = data.train.query_range(qid)
  click_prob = obs_prob[inv_ranking]*rel_train_prob[s_i:e_i]
  clicks = clk.bernoilli_sample_from_probs(click_prob)
  total_clicks += np.sum(clicks)

  if np.any(clicks):
    if prop_computed[qid] == False or np.any(np.equal(all_prop_scores[s_i:e_i][clicks], 0.)):
      (_, _, _,
       prob_per_rank) = pl.sample_rankings_query(
                      data.train,
                      qid,
                      policy_scores,
                      10**4,
                      cutoff=cutoff,
                      prob_per_rank=True,
                    )
      prop_computed[qid] = True
      all_prop_scores[s_i:e_i] = np.sum(prob_per_rank*obs_prob[:prob_per_rank.shape[0], None], axis=0)

    estimates[sample_i] = np.sum(clicks*train_doc_value_diff[s_i:e_i]/all_prop_scores[s_i:e_i])

  # if (sample_i+1) % 1000 == 0:
  if (sample_i+1) == measure_points[0]:
    measure_points = measure_points[1:]
    
    estimate = np.mean(estimates[:sample_i+1], dtype=np.float64)
    bin_estimate = np.sign(np.sum(estimates[:sample_i+1], dtype=np.float64))

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
        'logging policy CTR': total_clicks/np.float64(sample_i+1),
      }
    )

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(results, f)
