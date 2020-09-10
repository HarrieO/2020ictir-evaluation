# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.clicks as clk

def oracle_doc_variance(
          expected_reward,
          doc_values,
          rel_prob,
          obs_prob,
          sampled_inv_rankings):

  n_docs = rel_prob.shape[0]

  doc_obs_prob = np.mean(obs_prob[sampled_inv_rankings], axis=0)

  doc_click_prob = doc_obs_prob*rel_prob

  doc_score = n_docs * doc_values / doc_obs_prob
  doc_error = doc_score - expected_reward

  doc_variance = doc_click_prob*doc_error**2 + (1.-doc_click_prob)*expected_reward**2

  variance = np.mean(doc_variance)

  var_grad = rel_prob*(doc_error**2. - expected_reward**2. - 2.*doc_score*doc_error)

  return variance, var_grad


def oracle_list_variance(
          expected_reward,
          doc_values,
          rel_prob,
          obs_prob,
          doc_prop_scores,
          policy_log_scores,
          sampled_rankings,
          sampled_inv_rankings,
          sampled_ranking_probs,
          cutoff=None,
          compute_gradient=True):

  n_docs = rel_prob.shape[0]

  doc_click_prob = obs_prob[sampled_inv_rankings]*rel_prob[None, :]

  n_samples = sampled_rankings.shape[0]
  sampled_clicks = clk.bernoilli_sample_from_probs(doc_click_prob)

  doc_score = doc_values/doc_prop_scores

  click_values = sampled_clicks*doc_score[None, :]
  click_seq_values = np.sum(click_values, axis=1)

  click_seq_diff = (expected_reward - click_seq_values)
  click_seq_error = click_seq_diff**2.

  variance = np.mean(click_seq_error)  

  if not compute_gradient:
    return variance, None, None

  ind = np.arange(n_samples)
  score_grad = np.zeros(n_docs)
  temp_grad = np.zeros(n_docs)

  log_scores = np.tile(policy_log_scores[None,:], (n_samples, 1))

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])

    temp_grad[:] = 0.
    np.add.at(temp_grad, sampled_rankings[:, i], click_seq_error)
    score_grad += temp_grad/float(n_samples)
    score_grad -= np.mean(probs*click_seq_error[:, None], axis=0)

    log_scores[ind, sampled_rankings[:, i]] = np.NINF

  score_grad /= n_samples

  policy_grad = np.mean(
    2.*click_seq_diff[:,None]*sampled_clicks*doc_score[None,:]/doc_prop_scores[None,:],
    axis=0)

  return variance, score_grad, policy_grad


def oracle_data_split_list_variance(
                      data_split,
                      sample_ranking_f,
                      expected_reward,
                      doc_values,
                      rel_prob,
                      obs_prob,
                      policy_log_scores,
                      cutoff=None
                    ):
  mean_variance = 0.
  for qid in range(data_split.num_queries()):

    (sampled_rankings,
     sampled_inv_rankings,
     sampled_ranking_prob,
     prob_per_rank) = sample_ranking_f(qid)

    doc_prop_scores = np.sum(prob_per_rank*obs_prob[:prob_per_rank.shape[0], None], axis=0)

    s_i, e_i = data_split.query_range(qid)
    q_variance = oracle_list_variance(
                    expected_reward,
                    doc_values[s_i:e_i],
                    rel_prob[s_i:e_i],
                    obs_prob,
                    doc_prop_scores,
                    policy_log_scores[s_i:e_i],
                    sampled_rankings,
                    sampled_inv_rankings,
                    sampled_ranking_prob,
                    cutoff=cutoff,
                    compute_gradient=False,
                  )[0]
    mean_variance += q_variance
  return mean_variance/float(data_split.num_queries())


