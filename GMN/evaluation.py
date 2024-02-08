from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve

from GMN.loss import *


def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def cosine_similarity(x, y):
    return torch.sum(torch.mm(F.normalize(x), F.normalize(y).t()), dim=-1)


def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors
    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """
    if config['training']['loss'] == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config['training']['loss'] == 'hamming':
        return exact_hamming_similarity(x, y)
    elif config['training']['loss'] == 'cosine':
        return cosine_similarity(x, y)
    else:
        raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

    See `tf.metrics.auc` for more details about this metric.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1.
      labels: [n_examples] int.  Labels are either +1 or -1.
      **auc_args: other arguments that can be used by `tf.metrics.auc`.

    Returns:
      auc: the area under the ROC curve.
    """
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2
    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)


def auc_roc(df):
    Y = df.label.tolist()
    X = df.rel_score.tolist()

    y_true = [1 if x >= 2.5 else 0 for x in Y]
    y_pred = X
    roc = roc_auc_score(y_true, y_pred)
    print(roc)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, color="green", lw=2, label='ROC curve:2 set to 1, 1,0 set to 0 (area = %0.2f)' % (roc))

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AST Receiver Operating Characteristic (ROC) - Multi-class')
    plt.legend(loc='lower right')
    plt.show()


def auc_numpy(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

    See `tf.metrics.auc` for more details about this metric.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1.
      labels: [n_examples] int.  Labels are either +1 or -1.
      **auc_args: other arguments that can be used by `tf.metrics.auc`.

    Returns:
      auc: the area under the ROC curve.
    """
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    return metrics.auc(fpr, tpr)
