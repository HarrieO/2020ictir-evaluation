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
import utils.pretrained_models as prtr
import utils.EMPBM as empbm
import utils.logopt as logopt

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
parser.add_argument("--ranker_pair", type=int,
                    default=0,
                    help="Ranker pair to put in comparison.")
parser.add_argument("--neural", action='store_true',
                    help="Train a neural policy.")
parser.add_argument("--give_prop", action='store_true',
                    help="Give the true propensity scores.")

args = parser.parse_args()

click_model = args.click_model
binarize_labels = 'binarized' in click_model
eta = args.eta
cutoff = args.cutoff
give_prop = args.give_prop

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

results = {
  'model name': 'EM LogOpt',
  'model type': 'EM LogOpt',
  'click model': click_model,
  'dataset name': args.dataset,
  'fold': fold_id,
  'train difference': expected_train_reward,
  'validation difference': expected_vali_reward,
  'test difference': test_diff,
  'results': [],
  }
if give_prop:
  results['model name'] = 'EM Given LogOpt'
  results['model type'] = 'EM Given LogOpt'

n_feat = data.num_features + additional_train_feat.shape[1]
if args.neural:
  import models.neural as dnnm
  logging_policy = dnnm.NeuralModel(n_feat)
else:
  logging_policy = lnm.LinearModel(n_feat) 

pos_bias = np.zeros(obs_prob.shape, dtype=np.float64)
pos_bias[1] = 1.
rel_est = np.full(data.train.num_docs(), 0.1, dtype=np.float64)
logopt.optimize_logging_policy( 1000,
                                logging_policy,
                                data.train,
                                additional_train_feat,
                                train_doc_value_diff,
                                cutoff,
                                pos_bias,
                                rel_est,
                                0.01)

policy_scores = logging_policy.score_data_split(
                          data.train,
                          additional_feat=additional_train_feat,
                        )

n_queries_to_sample = 3*10**6
measure_points = np.unique(np.array(
                sorted(list(np.unique(
                  np.geomspace(10**2, n_queries_to_sample, 500).astype(np.int64))) 
                      + [10**x for x in range(2,7)])))
EM_points = np.array([10**x for x in range(2, 7)])

doc_freq = np.zeros(data.train.num_docs(), dtype=np.int64)
clicks_per_doc = np.zeros(data.train.num_docs(), dtype=np.int64)
clicks_per_doc_rank = np.zeros((data.train.num_docs(), cutoff+1), dtype=np.int64)
doc_rank_freq = np.zeros((data.train.num_docs(), cutoff+1), dtype=np.int64)
update_prop = np.zeros(data.train.num_docs(), dtype=bool)
total_clicks = 0
click_ids = np.zeros(n_queries_to_sample*cutoff, dtype=np.int32)
sample_ids = np.zeros(n_queries_to_sample*cutoff, dtype=np.int32)
estimates = np.zeros(n_queries_to_sample, dtype=np.float64)
all_prop_scores = np.ones(data.train.num_docs(), dtype=np.float64)
doc_weighted_values = np.zeros(data.train.num_docs(), dtype=np.float64)
for sample_i in range(n_queries_to_sample):
  qid = np.random.choice(data.train.num_queries())
  q_n_docs = data.train.query_size(qid)

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

  clicks_per_doc[s_i:e_i] += clicks
  doc_freq[s_i:e_i] += 1
  doc_rank_freq[s_i + np.arange(q_n_docs), inv_ranking] += 1
  update_prop[s_i:e_i] = np.greater(clicks_per_doc[s_i:e_i], 0)

  if np.any(clicks):
    prev_total = total_clicks
    total_clicks += np.sum(clicks)
    click_ids[prev_total:total_clicks] = s_i + np.where(clicks)[0]
    sample_ids[prev_total:total_clicks] = sample_i

    clicks_per_doc_rank[s_i + np.arange(q_n_docs),
                           inv_ranking] += clicks

  if (sample_i+1) == measure_points[0]:
    measure_points = measure_points[1:]
    
    if (sample_i+1) in EM_points:
      # print('estimating bias')
      if give_prop:
        pos_est, rel_est = empbm.train_click_model(data.train,
                                                   doc_rank_freq[:,:-1],
                                                   clicks_per_doc_rank[:,:-1],
                                                   pos_bias=obs_prob[:cutoff])
      else:
        pos_est, rel_est = empbm.train_click_model(data.train,
                                                   doc_rank_freq[:,:-1],
                                                   clicks_per_doc_rank[:,:-1])
      pos_bias[:cutoff] = pos_est
      pos_bias[:cutoff] = np.maximum(pos_bias[:cutoff], 0.01)
      print(sample_i+1, 'Position Bias Estimate:', pos_bias[:cutoff])
      # pos_bias = obs_prob[:cutoff]
      update_prop[:] = np.greater(clicks_per_doc, 0)
      # estimates[:sample_i] = 0.

    update_prob_per_rank = (doc_rank_freq[update_prop].astype(np.float64)
                            /np.maximum(1.,
                              np.sum(doc_rank_freq, axis=1)[update_prop][:, None]))
    all_prop_scores[update_prop] = np.dot(update_prob_per_rank[:,:cutoff], pos_bias[:cutoff])
    doc_weighted_values[update_prop] = train_doc_value_diff[update_prop]/all_prop_scores[update_prop]

    # click_i = 0
    # estimates[:sample_i] = 0.
    # all_prob_per_rank = doc_rank_freq.astype(np.float64)/np.maximum(1., np.sum(doc_rank_freq, axis=1)[:, None])
    # all_prop_scores = np.dot(all_prob_per_rank[:,:cutoff], pos_bias)

    click_mask = update_prop[click_ids[:total_clicks]]
    cur_click_ids = click_ids[:total_clicks][click_mask]
    masked_sample_ids = sample_ids[:total_clicks][click_mask]
    estimates[masked_sample_ids] = 0.
    np.add.at(estimates, masked_sample_ids, doc_weighted_values[cur_click_ids])
    update_prop[:] = False

    # print('Updating', np.sum(click_mask)/float(total_clicks))

    click_i = total_clicks

    # print(click_ids[:total_clicks])
    # print(sample_ids[:total_clicks])
    # estimates[:sample_i] = 0.
    # for i in range(total_clicks):
    #   c_i = click_ids[i]
    #   estimates[sample_ids[i]] += train_doc_value_diff[c_i]/all_prop_scores[c_i]

    # print(estimates[:sample_i+1])
    estimate = np.mean(estimates[:sample_i+1], dtype=np.float64)
    bin_estimate = np.sign(np.sum(estimates[:sample_i+1], dtype=np.float64))

    # print('Estimate from %d queries, estimate: %0.05f squared error: %0.09f error: %0.09f' % (
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

    if (sample_i+1) in EM_points:
      print(sample_i+1, 'Updating logging policy')
      logopt.optimize_logging_policy(
                                1000,
                                logging_policy,
                                data.train,
                                additional_train_feat,
                                train_doc_value_diff,
                                cutoff,
                                pos_bias,
                                rel_est,
                                estimate)

      policy_scores = logging_policy.score_data_split(
                                data.train,
                                additional_feat=additional_train_feat,
                              )
      print('done')

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(results, f)
