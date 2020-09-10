import numpy as np
import tensorflow as tf

class NeuralModel(object):

  def __init__(self, num_features, n_layers=2, n_units=32):
    layers = []
    layers += [tf.keras.layers.InputLayer(input_shape=num_features, dtype='float64')]
    layers += [tf.keras.layers.Dense(n_units, activation='sigmoid', dtype='float64')
               for _ in range(n_layers)]
    layers += [tf.keras.layers.Dense(1, dtype='float64')]
    self.model = tf.keras.models.Sequential(layers)
    # self.optimizer = tf.keras.optimizers.SGD(learning_rate=1.)
    self.optimizer = tf.keras.optimizers.Adam()

  def score(self, feature_matrix):
    return self.model(feature_matrix).numpy()[:, 0]

  def score_query(self, qid, data_split, additional_feat=None):
    query_feat = data_split.query_feat(qid)
    if additional_feat is not None:
      s_i, e_i = data_split.query_range(qid)
      query_feat = np.concatenate(
                        (query_feat,
                        additional_feat[s_i:e_i, :]),
                        axis=1,
                      )
    return self.score(query_feat)

  def score_data_split(self, data_split, additional_feat=None):
    data_feat = data_split.feature_matrix
    if additional_feat is not None:
      data_feat = np.concatenate(
                        (data_feat,
                        additional_feat),
                        axis=1,
                      )
    return self.score(data_feat)

  def gradient_update(self, doc_weights, doc_feat,
                      additional_feat=None,
                      learning_rate=0.01):
    doc_weights = np.nan_to_num(doc_weights)
    feat_input = doc_feat
    if additional_feat is not None:
      feat_input = np.concatenate(
                        (feat_input, additional_feat),
                        axis=1,
                      )

    with tf.GradientTape() as tape:
      predictions = self.model(feat_input)
      loss = -tf.math.reduce_sum(predictions[:,0]*doc_weights)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

  def predict(self, feat):
    logits = self.model(feat)
    return tf.sigmoid(logits).numpy()[:, 0]

  def log_likelihood_update(self, feat, targets):

    with tf.GradientTape() as tape:
      logits = self.model(feat)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
                      labels=targets, logits=logits,
                    )

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
