import numpy as np
import models.neural as dnnm

def train_click_model(data_split, display, clicks, pos_bias=None):
  n_total_display = np.sum(display)
  n_display_per_d = np.sum(display, axis=1)
  doc_mask = np.greater(n_display_per_d, 0)
  n_doc_included = np.sum(doc_mask, dtype=np.int64)

  if pos_bias is None:
    pos_known = False
  else:
    pos_known = True
  
  doc_weight = n_display_per_d/np.sum(n_display_per_d, dtype=np.float64)
  doc_weight = doc_weight[doc_mask]

  doc_ids = np.arange(display.shape[0])[doc_mask]
  doc_clicks = clicks[doc_mask]
  doc_display = display[doc_mask]

  ctr_per_pos = np.sum(doc_clicks, axis=0, dtype=np.float64)/np.sum(doc_display, axis=0, dtype=np.float64)

  doc_rel = np.full(n_doc_included, ctr_per_pos[0], dtype=np.float64)
  if not pos_known:
    pos_bias = ctr_per_pos
    if pos_bias[0] > 0:
      pos_bias /= pos_bias[0]

  n_per_rank = np.sum(doc_display, axis=0, dtype=np.float64)
  n_per_doc = np.sum(doc_display, axis=1, dtype=np.float64)

  doc_non_clicks = doc_display - doc_clicks

  rel_model = dnnm.NeuralModel(data_split.datafold.num_features)

  base_steps = int(np.log10(n_total_display))
  for i in range(100):
    denom = (1.-pos_bias[None, :]*doc_rel[:, None])
    denom[np.equal(denom, 0.)] = 1.
    # E1R1 = doc_clicks.copy()
    if not pos_known:
      E1R0 = pos_bias[None, :]*(1.-doc_rel[:, None])/denom
    E0R1 = (1.-pos_bias[None, :])*doc_rel[:, None]/denom
    # E0R0 = (1.-pos_bias[None, :])*(1.-doc_rel[:, None])/denom

    if not pos_known:
      pos_bias = np.sum(E1R0*doc_non_clicks + doc_clicks,
                      axis=0, dtype=np.float64)/n_per_rank
    doc_rel = np.sum(E0R1*doc_non_clicks + doc_clicks,
                     axis=1, dtype=np.float64)/n_per_doc

    if not pos_known:
      if pos_bias[0] > 0:
        pos_bias /= pos_bias[0]

    for _ in range(75):
      sampled_i = np.random.choice(n_doc_included,
                                   size=512,
                                   p=doc_weight,
                                   replace=True)
      target = doc_rel[sampled_i]
      feat = data_split.feature_matrix[sampled_i, :]

      rel_model.log_likelihood_update(feat, target[:, None])

    doc_rel = rel_model.predict(data_split.feature_matrix[doc_ids,:])

    # print(i, pos_bias)
    # print(np.mean(doc_rel))

  # if base_steps > 6:
  #   for _ in range(base_steps**3):
  #     sampled_i = np.random.choice(n_doc_included,
  #                                  size=512,
  #                                  p=doc_weight,
  #                                  replace=True)
  #     target = doc_rel[sampled_i]
  #     feat = data_split.feature_matrix[sampled_i, :]

  #     rel_model.log_likelihood_update(feat, target[:, None])

  #   doc_rel = rel_model.predict(data_split.feature_matrix[doc_ids,:])

  all_doc_rel = rel_model.predict(data_split.feature_matrix)
  return pos_bias, all_doc_rel
