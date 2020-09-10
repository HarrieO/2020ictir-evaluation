# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import policies.plackettluce as pl
import utils.variance as var

def optimize_logging_policy(n_update_steps,
                            logging_policy,
                            data_split,
                            additional_feat,
                            doc_value_diff,
                            cutoff,
                            obs_prob,
                            rel_prob,
                            expected_value):

  update_i = 0
  for update_i in range(n_update_steps):
    qid = np.random.choice(data_split.num_queries())
    q_n_docs = data_split.query_size(qid)
    s_i, e_i = data_split.query_range(qid)
    q_value_diff = doc_value_diff[s_i:e_i]

    update_i += 1
    if np.all(np.equal(q_value_diff, 0.)):
      continue

    policy_query_scores = logging_policy.score_query(
                              qid,
                              data_split,
                              additional_feat=additional_feat,
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
                                          expected_value,
                                          doc_value_diff[s_i:e_i],
                                          rel_prob[s_i:e_i],
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
                      data_split.query_feat(qid),
                      learning_rate=10**-2,
                      additional_feat=additional_feat[s_i:e_i, :],
                    )