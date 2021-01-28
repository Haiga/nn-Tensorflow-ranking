import os
import tensorflow as tf
import tensorflow_core
import tensorflow_ranking as tfr
# from tensorflow_core.python.data.ops.dataset_ops import DatasetV1

from utils.dataset import getParamsByDataset, getData
from utils.metrics import mNdcg
import numpy as np
import warnings

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

_TRAIN_DATA_PATH, _TEST_DATA_PATH, _VALI_DATA_PATH, _LIST_SIZE, _NUM_FEATURES = \
    getParamsByDataset("2003_td_dataset")

tf.enable_eager_execution()
tf.executing_eagerly()

# Store the paths to files containing training and test instances.
# As noted above, we will assume the data is in the LibSVM format
# and that the content of each file is sorted by query ID.
# _TRAIN_DATA_PATH = "/tmp/train.txt"
# _TEST_DATA_PATH = "/tmp/test.txt"

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.
_LOSS = "pairwise_logistic_loss"

# In the TF-Ranking framework, a training instance is represented
# by a Tensor that contains features from a list of documents
# associated with a single query. For simplicity, we fix the shape
# of these Tensors to a maximum list size and call it "list_size,"
# the maximum number of documents per query in the dataset.
# In this demo, we take the following approach:
#   * If a query has fewer documents, its Tensor will be padded
#     appropriately.
#   * If a query has more documents, we shuffle its list of
#     documents and trim the list down to the prescribed list_size.
# _LIST_SIZE = 100

# The total number of features per query-document pair.
# We set this number to the number of features in the MSLR-Web30K
# dataset.
# _NUM_FEATURES = 136

# Parameters to the scoring function.
_BATCH_SIZE = 4
_HIDDEN_LAYER_DIMS = ["20", "10"]

"""### Input Pipeline

The first step to construct an input pipeline that reads your dataset and produces a `tensorflow.data.Dataset` object. In this example, we will invoke a LibSVM parser that is included in the `tensorflow_ranking.data` module to generate a `Dataset` from a given file.

We parameterize this function by a `path` argument so that the function can be used to read both training and test data files.
"""


def input_fn(path):
    train_dataset = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(path, _NUM_FEATURES, _LIST_SIZE),
        output_types=(
            {str(k): tf.float32 for k in range(1, _NUM_FEATURES + 1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([_LIST_SIZE, 1])
             for k in range(1, _NUM_FEATURES + 1)},
            tf.TensorShape([_LIST_SIZE])
        )
    )

    train_dataset = train_dataset.shuffle(1000).repeat().batch(_BATCH_SIZE)
    return train_dataset.make_one_shot_iterator().get_next()


from utils.generator import libsvm_generator


def input_fn_predict(path):
    from utils.generator import libsvm_generator

    train_dataset = tf.data.Dataset.from_generator(
        # train_dataset = tensorflow_core.python.data.ops.dataset_ops.DatasetV1.from_generator(
        # tfr.data.libsvm_generator(path, _NUM_FEATURES, 1000),
        libsvm_generator(path, _NUM_FEATURES, 1000, False),
        output_types=(
            {str(k): tf.float32 for k in range(1, _NUM_FEATURES + 1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([1000, 1])
             for k in range(1, _NUM_FEATURES + 1)},
            tf.TensorShape([1000])
        )
    )
    # train_dataset = train_dataset.shuffle(1000).repeat().batch(10)
    # train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=False).repeat(1).batch(10)
    # train_dataset = train_dataset.batch(10)
    return train_dataset.batch(10).make_one_shot_iterator().get_next()
    # return tf.data.make_one_shot_iterator(train_dataset).get_next()
    # return train_dataset


"""### Scoring Function

Next, we turn to the scoring function which is arguably at the heart of a TF Ranking model. The idea is to compute a relevance score for a (set of) query-document pair(s). The TF-Ranking model will use training data to learn this function.

Here we formulate a scoring function using a feed forward network. The function takes the features of a single example (i.e., query-document pair) and produces a relevance score.
"""


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = [
        "%d" % (i + 1) for i in range(0, _NUM_FEATURES)
    ]
    return {
        name: tf.feature_column.numeric_column(
            name, shape=(1,), default_value=0.0) for name in feature_names
    }


def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer_width,
                activation="tanh")

        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


"""### Evaluation Metrics

We have provided an implementation of popular Information Retrieval evalution metrics in the TF Ranking library.
"""


def eval_metric_fns():
    """Returns a dict from name to metric functions.

    This can be customized as follows. Care must be taken when handling padded
    lists.

    # def _auc(labels, predictions, features):
    #   is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    #   clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    #   clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    #   return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
    # metric_fns["auc"] = _auc

    Returns:
      A dict mapping from metric name to a metric function with above signature.
    """
    metric_fns = {}

    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10]
    })

    return metric_fns


"""### Putting It All Together

We are now ready to put all of the components above together and create an `Estimator` that can be used to train and evaluate a model.
"""


def get_estimator(hparams):
    """Create a ranking estimator.

    Args:
      hparams: (tf.contrib.training.HParams) a hyperparameters object.

    Returns:
      tf.learn `Estimator`.
    """

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer="Adagrad")

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(_LOSS),
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=_train_op_fn)

    return tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=1,
            transform_fn=None,
            ranking_head=ranking_head),
        params=hparams)


"""Let us instantiate and initialize the `Estimator` we defined above."""

hparams = tf.contrib.training.HParams(learning_rate=0.05)
ranker = get_estimator(hparams)

"""Now that we have a correctly initialized `Estimator`, we will train a model using the training data. We encourage you to experiment with different number of steps here and below."""

# ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=10)

"""Finally, let us evaluate our model on the test set."""

# r = ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH), steps=10)
# for item in input_fn_predict(_TEST_DATA_PATH).make_one_shot_iterator().get_next():
# for item in input_fn_predict(_TEST_DATA_PATH):
#     xi, yi = item
#
predictions_iterator = ranker.predict(input_fn=lambda: input_fn_predict(_TEST_DATA_PATH))
r = list(predictions_iterator)
p = input_fn_predict(_TEST_DATA_PATH)
# pred_fn, pred_hook = get_eval_inputs(input_fn_predict)
# generator_ = ranker.predict(input_fn=pred_fn, hooks=[pred_hook])
# pred_list = list(generator_)

x = 2
# all_predictions = []
# with open("out.txt", "w") as fo:
#     for i in range(10):
#         array_of_predictions = next(predictions_iterator)
#         all_predictions.append(array_of_predictions)
#         # for k in array_of_predictions:
#         #     fo.write(str(k) + "\n")
# # print(r)
# print(predictions_iterator)
#
# X, y, queries_id, y_true_by_query = getData(_TEST_DATA_PATH)
#
# ndcgs = mNdcg(y_true_by_query, all_predictions)
# print("----")
# print(np.mean(ndcgs))
# # print(mNdcg(input_fn_predict(_TEST_DATA_PATH)[1].numpy(), all_predictions))
# # print("----")
# # print(np.sum(input_fn_predict(_TEST_DATA_PATH)[1].numpy()[0]))
# # print("----")
# # print(input_fn_predict(_TEST_DATA_PATH)[1].numpy()[1][0])
# # print(input_fn_predict(_TEST_DATA_PATH)[1].numpy()[1][1])
# # print(input_fn_predict(_TEST_DATA_PATH)[1].numpy()[1][2])
# # print(input_fn_predict(_TEST_DATA_PATH)[1].numpy()[1][3])
# print("----")
# # print("fim")
