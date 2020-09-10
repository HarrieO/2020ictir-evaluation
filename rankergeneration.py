# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import random
import utils.dataset as dataset
import utils.ranking as rnk
import utils.clicks as clk

parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str,
                    help="Path to file for pretrained model.")
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--num_queries", type=int,
                    default=100,
                    help="Number of queries to take from training/validation set.")
parser.add_argument("--num_rankers", type=int,
                    default=100,
                    help="Number of rankers to generate.")
parser.add_argument("--eta", type=float,
                    default=1.0,
                    help="Eta parameter for observance probabilities.")
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='linear1.0')
parser.add_argument("--cutoff", type=int,
                    default=10,
                    help="Number of documents that are displayed.")
parser.add_argument("--min_diff", type=float,
                    help="Minimal difference in click through percentage.",
                    default=0.00)
args = parser.parse_args()
num_select_queries = args.num_queries

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                )

data = data.get_data_folds()[0]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

obs_prob = clk.inverse_rank_prob(
                    np.arange(data.validation.max_query_size(), dtype=np.float64),
                    args.eta
                  )
if args.cutoff:
  obs_prob[args.cutoff:] = 0
rel_prob_f = clk.get_relevance_click_model(args.click_model)

def calc_true_loss(ranking_model, data_split):
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)
  _, inv_rankings = rnk.data_split_rank_and_invert(all_scores, data_split)
  click_prob = obs_prob[inv_rankings]*rel_prob_f(data_split.label_vector)
  result = np.mean(click_prob, dtype=np.float64)
  result *= data_split.num_docs()/float(data_split.num_queries())
  return -result

def calc_sub_loss(ranking_model, data_split, query_selection):
  selected_results = np.zeros(query_selection.shape, dtype=np.float64)
  for i, qid in enumerate(query_selection):
    s_i, e_i = data_split.query_range(qid)
    feat = data_split.query_feat(qid)
    scores = np.dot(feat, ranking_model)
    inv_ranking = rnk.rank_and_invert(scores)[1]
    # click_prob = obs_prob[inv_ranking]*rel_prob_f(data_split.query_labels(qid))
    # selected_results[i] = np.sum(click_prob, dtype=np.float64)
    nom = 2.**data_split.query_labels(qid) - 1.
    denom = np.log2(inv_ranking.astype(np.float64)+2.)
    selected_results[i] = np.sum(nom/denom, dtype=np.float64)
  return -np.mean(selected_results, dtype=np.float64)

def optimize(data,
             train_queries,
             validation_queries,
             mask,
             learning_rate,
             cutoff=10,
             trial_epochs=3,
             max_epochs=200,
             epsilon_thres=0.001,
             learning_rate_decay=0.9999):
  starting_learning_rate = learning_rate

  train_data = data.train
  vali_data = data.validation

  train_rel_prob = rel_prob_f(data.train.label_vector)
  train_rel_nom = 2.**data.train.label_vector - 1.

  best_model = np.zeros(train_data.datafold.num_features)
  best_loss = np.inf
  pivot_loss = np.inf
  # model = np.zeros(train_data.datafold.num_features)
  model = np.random.normal(size=train_data.datafold.num_features)

  start_time = time.time()

  epoch_i = 0
  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    q_permutation = np.random.permutation(train_queries)
    for qid in q_permutation:
      q_docs = data.train.query_feat(qid)
      q_scores = np.dot(q_docs, model)
      n_docs = q_docs.shape[0]

      s_i, e_i = data.train.doclist_ranges[qid:qid+2]
      # q_prop = train_rel_prob[s_i:e_i]
      q_prop = train_rel_nom[s_i:e_i]

      q_inv = rnk.rank_and_invert(q_scores)[1]

      prop_diff = q_prop[:, None] - q_prop[None, :]
      prop_mask = np.less_equal(prop_diff, 0.)

      rnk_vec = np.less(q_inv, cutoff)
      rnk_mask = np.logical_or(rnk_vec[:, None],
                                rnk_vec[None, :])

      prop_mask = np.logical_or(np.logical_not(rnk_mask), prop_mask)

      rank_diff = np.abs(q_inv[:, None] - q_inv[None, :])
      rank_diff[prop_mask] = 1.

      disc_upp = 1. / np.log2(rank_diff+1.)
      disc_low = 1. / np.log2(rank_diff+2.)
      disc_upp[np.greater(rank_diff, cutoff)] = 0.
      disc_low[np.greater(rank_diff, cutoff-1)] = 0.

      pair_w = disc_upp - disc_low
      pair_w *= np.abs(prop_diff)
      pair_w[prop_mask] = 0.
      
      score_diff = q_scores[:, None] - q_scores[None, :]
      score_diff[prop_mask] = 0.
      safe_diff = np.minimum(-score_diff, 500)
      act = 1./(1 + np.exp(safe_diff))
      act[prop_mask] = 0.
      safe_exp = pair_w - 1.
      safe_exp[prop_mask] = 0.

      log2_grad = 1./(act**pair_w*np.log(2))
      power_grad = pair_w*(act)**safe_exp
      sig_grad = act*(1-act)

      activation_gradient = -log2_grad*power_grad*sig_grad

      np.fill_diagonal(activation_gradient,
                         np.diag(activation_gradient)
                         - np.sum(activation_gradient, axis=1))

      doc_weights = np.sum(activation_gradient, axis=0)


      gradient = np.sum(q_docs * doc_weights[:, None], axis=0)

      model = model + learning_rate*gradient*mask
      learning_rate *= learning_rate_decay


    epoch_i += 1
    cur_loss = calc_sub_loss(model, vali_data, validation_queries)
    # print(epoch_i, '%0.05f' % cur_loss, cur_loss - best_loss)
    if cur_loss < pivot_loss:
      best_model = model[:]
      best_loss = cur_loss
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

    # train_loss = calc_sub_loss(model, train_data, train_queries)
  true_loss = calc_true_loss(best_model,
                                vali_data)
    # print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % train_loss, '%0.05f' % true_loss,)
    # print(epoch_i, '%0.05f' % train_loss, '%0.05f' % cur_loss, '%0.05f' % true_loss,)

  # print(epoch_i, starting_learning_rate, '%0.06f' % learning_rate_decay, '%0.05f' % best_loss, '%0.05f' % true_loss)

  result = {
      'model': best_model,
      'estimated_loss': best_loss,
      'true_loss': true_loss,
      'epoch': epoch_i - trial_epochs,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': starting_learning_rate,
      'learning_rate_decay': learning_rate_decay,
      'trial_epochs': trial_epochs,
      'num_queries_sampled': num_select_queries,
    }
  return result

def train_ranker(seed):

  train_q = np.random.permutation(data.train.num_queries())
  train_q = train_q[:num_select_queries + 1]
  vali_q = np.random.permutation(data.validation.num_queries())
  vali_q = vali_q[:num_select_queries + 1]

  mask = np.zeros(data.num_features, dtype=np.float64)
  selected = np.random.permutation(
                np.arange(data.num_features)
              )[:int(data.num_features*0.5)]
  mask[selected] = 1.

  result = optimize(data,
                    train_q,
                    vali_q,
                    mask=mask,
                    learning_rate=0.01,
                    trial_epochs=10,
                    learning_rate_decay=0.99)

  def _doc_feat_str(doc_feat):
    doc_str = ""
    for f_i, f_v in enumerate(doc_feat):
      if f_v == 1.:
        doc_str += ' %d:1' % data.feature_map[f_i]
      elif f_v != 0.:
        doc_str += ' %d:%f' % (data.feature_map[f_i], f_v)
    return doc_str

  return -result['true_loss'], _doc_feat_str(result['model'])

results = [train_ranker(x) for x in np.random.uniform(
                                            args.num_rankers*100,
                                            size=args.num_rankers)]
random.shuffle(results)

ctr_list = np.array([x[0] for x in results])
model_list = [x[1] for x in results]

print('CTR:')
print(sorted(ctr_list)[::-1])

print('Diff:')
print(sorted(np.abs(ctr_list[::2] - ctr_list[1::2]))[::-1])

output = '\n'.join(model_list)
with open(args.output_path, 'w') as f:
  f.write(output)



