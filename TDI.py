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
  'model name': 'Team-Draft Interleaving',
  'model type': 'Team-Draft Interleaving',
  'click model': click_model,
  'train difference': expected_train_reward,
  'validation difference': expected_vali_reward,
  'test difference': test_diff,
  'dataset name': args.dataset,
  'fold': fold_id,
  'results': [],
    # {
    #   'num queries': 0,
    #   'binary error': 0.5,
    #   'binary train error': 0.5,
    #   'squared error': 0.5,
    #   'absolute error': 0.5,
    # }]
  }

n_queries_to_sample = 3*10**6
measure_points = np.unique(np.array(
                sorted(list(np.unique(
                  np.geomspace(10**2, n_queries_to_sample, 500).astype(np.int64))) 
                      + [10**x for x in range(2,7)])))

total_clicks = 0
n_models = models_train_rankings.shape[0]
estimates = np.zeros(n_queries_to_sample, dtype=np.float64)
for sample_i in range(n_queries_to_sample):
  qid = np.random.choice(data.train.num_queries())
  q_n_docs = data.train.query_size(qid)
  ranking_len = np.minimum(q_n_docs, cutoff+1)
  s_i, e_i = data.train.query_range(qid)

  cur_rankings = models_train_rankings[:, s_i:e_i]
  interleaving = np.zeros(ranking_len, dtype=np.int32)
  assignment = np.zeros(ranking_len, dtype=np.float64)
  i = 0
  while i < ranking_len and  cur_rankings[0, i] == cur_rankings[1, i]:
    interleaving[i] = cur_rankings[0, i]
    i += 1

  assignment[i::2] = np.random.choice(2, size=assignment[i::2].shape)*2-1
  assignment[i+1::2] = -1*assignment[i:ranking_len-1:2]
  i_0 = i
  i_1 = i
  for i in range(i, ranking_len):
    if assignment[i] == 1:
      while np.any(np.equal(cur_rankings[0, i_0],interleaving[:i])):
        i_0 += 1
      interleaving[i] = cur_rankings[0, i_0]
    else:
      while np.any(np.equal(cur_rankings[1, i_1],interleaving[:i])):
        i_1 += 1
      interleaving[i] = cur_rankings[1, i_1]
    i+=1

  cur_rel_probs = rel_train_prob[s_i:e_i]
  click_prob = obs_prob[:ranking_len]*cur_rel_probs[interleaving]
  clicks = clk.bernoilli_sample_from_probs(click_prob)

  total_clicks += np.sum(clicks)

  estimates[sample_i] = np.sign(np.sum(clicks*assignment))

  # if (sample_i+1) % 1000 == 0:
  if (sample_i+1) == measure_points[0]:
    measure_points = measure_points[1:]

    estimate = np.mean(estimates[:sample_i])
    bin_estimate = np.sign(np.sum(estimates[:sample_i]))

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
