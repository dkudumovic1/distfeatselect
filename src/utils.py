import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_recall_curve, \
    auc, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC


def create_balanced_distributions(labels: np.ndarray, n_feats: int, n_vbins: int, n_hbins: int):
    """
    Combines outputs from feature_distribution and balanced_sample_partition to create class-balanced feature and
    sample partitions for the dataset.

    Parameters
    ----------
    labels : np.ndarray
        Data labels.
    n_feats: int
        Number of features in the data.
    n_vbins : int
        Number of feature partitions of the data.
    n_hbins : int
        Number of sample partitions of the data.

    Returns
    -------
    tuple
        A tuple containing the class-balanced feature partitions and sample partitions for the dataset.
    """

    return feature_distribution(n_feats=n_feats, n_vbins=n_vbins
                                ), \
        balanced_sample_partition(labels=labels, n_hbins=n_hbins
                                  )


def balanced_sample_partition(labels: np.ndarray, n_hbins: int) -> np.ndarray:
    """
    Creates class-balanced sample partitions for the dataset.

    Parameters
    ----------
    labels : np.ndarray
        Data labels.
    n_hbins : int
        Number of sample partitions of the data.

    Returns
    -------
    sample_partitions : np.ndarray
        Class balanced version of sample_distribution.
    """
    index_by_label = {
        l: np.where(labels == l)[0] for l in np.unique(labels)
    }

    index_by_label_split = {
        key: sample_distribution(n_samples=None, n_hbins=n_hbins, sample=value) for key, value in index_by_label.items()
    }

    _partitions = []
    for i in range(n_hbins):
        combined_sample = np.concatenate(
            [index_by_label_split[key][:, i] for key in index_by_label_split],
            axis=0,
        )
        _partitions.append(random.sample(list(combined_sample), len(combined_sample)))

    return np.transpose(np.array(_partitions))


def sample_distribution(n_samples: int, n_hbins: int, sample: np.ndarray = None) -> np.ndarray:
    """
    Creates sample bins for the dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples in the data.
    n_hbins : int
        Number of sample partitions of the data.
    sample : np.ndarray, optional
        If sample is not None, then the sample indexes are taken from sample, otherwise they are generated from
        np.arange(n_samples). sample argument required for creating balanced sample partitions.

    Returns
    -------
    sample_partitions : np.ndarray
        Contains in each column the sample indexes of the samples that belong to that sample bin.
    """
    if sample is None:
        sample_index = np.arange(n_samples).tolist()
    else:
        sample_index = sample.tolist()
        n_samples = len(sample_index)
    rnd_list = random.sample(sample_index, len(sample_index))  # random shuffle of feature ids

    _dups = (n_hbins - (len(rnd_list) % n_hbins)) if (n_samples % n_hbins != 0) else 0  # number of duplicates to add

    _comb = rnd_list + random.sample(sample_index, _dups)
    sample_partitions = np.reshape(_comb, (len(_comb) // n_hbins, n_hbins))
    return sample_partitions


def feature_distribution(n_feats: int, n_vbins: int) -> np.ndarray:
    """
    Function that creates feature bins for the features in the dataset.

    Parameters
    ----------
    n_feats : int
        Number of features in the data.
    n_vbins : int
        Number of feature partitions of the data.

    Returns
    -------
    feature_partitions : np.ndarray
        Contains in each column the features that belong to that feature bin.
    """

    feature_index = np.arange(1, n_feats).tolist()  # list of feature ids
    rnd_list = random.sample(feature_index, len(feature_index))  # random shuffle of feature ids

    _dups = (n_vbins - (len(rnd_list) % n_vbins)) if (
            (n_feats - 1) % n_vbins != 0) else 0  # number of duplicates to add

    _comb = rnd_list + random.sample(feature_index, _dups)
    rnd_mat = np.reshape(_comb, (len(_comb) // n_vbins, n_vbins))
    feature_partitions = np.vstack((np.zeros((n_vbins,)), rnd_mat))
    return feature_partitions


def evaluate_interim_model(
        model_features: list,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
        model=SVC(kernel='linear', random_state=42)
):
    """
    Evaluates the performance of a machine learning model on the validation set using cross-validation.

    Parameters
    ----------
    model_features : list
        Subset of features to be included in the model.
    X : np.ndarray
        Data. Feature matrix.
    y : np.ndarray
        Labels.
    metric : str {'acc', 'roc_auc', 'avg_prec', 'f1', 'auprc'}
        The evaluation metric to be used. Supported metrics: 'acc' (accuracy), 'roc_auc'
        (Receiver Operating Characteristic - Area Under the Curve), 'avg_prec' (average precision),
        'f1' (F1 score), 'au_prc' (Area Under the Precision-Recall Curve).
    model : object, optional
        The machine learning model to be evaluated. Default is a Support Vector Classifier (SVC)
        with a linear kernel and a random seed of 42.

    Returns
    -------
    model_final : object
        The best-fitted machine learning model based on cross-validation.
    performance : float
        Mean performance on the validation set based on the specified metric across all folds.
    """
    X = X[:, model_features]

    scoring = {'acc': make_scorer(accuracy_score),
               'roc_auc': make_scorer(roc_auc_score),
               'avg_prec': make_scorer(average_precision_score),
               'f1': make_scorer(f1_score, average='macro'),
               'au_prc': make_scorer(au_prc)
               }

    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    if model is None: model = SVC(kernel='linear', random_state=42)

    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False,
                                return_estimator=True)

    acc_scores = cv_results['test_acc']
    roc_auc_scores = cv_results['test_roc_auc']
    avg_prec_scores = cv_results['test_avg_prec']
    f1_scores = cv_results['test_f1']
    au_prc_scores = cv_results['test_au_prc']
    fitted_models = cv_results['estimator']

    if metric == 'acc':
        performance = np.mean(acc_scores)
    elif metric == 'roc_auc':
        performance = np.mean(roc_auc_scores)
    elif metric == 'avg_prec':
        performance = np.mean(avg_prec_scores)
    elif metric == 'f1':
        performance = np.mean(f1_scores)
    elif metric == 'au_prc':
        performance = np.mean(au_prc_scores)
    else:
        raise ValueError("Invalid metric. Supported metrics: 'acc', 'roc_auc', 'avg_prec', 'f1', 'au_prc'")

    return fitted_models[0], performance


def model_score(
        method: str,
        y_true: np.ndarray,
        y_pred_label: np.ndarray,
        y_pred_prob: np.ndarray
) -> float:
    """
    Evaluates model performance based on specified metric using sklearn.metrics.

    Parameters
    ----------
    method : str {'acc', 'roc_auc', 'avg_prec','f1', 'auprc'}
        Metric used to evaluate model performance.
    y_true : np.ndarray
        {0,1} ground truth labels.
    y_pred_label : np.ndarray
        {0,1} predicted labels.
    y_pred_prob : np.ndarray
        [0,1] predicted probabilities.

    Returns
    -------
    out : float
        Output based on metric.
    """
    methods = {
        'acc': accuracy_score(
            y_true=y_true,
            y_pred=y_pred_label
        ),
        'roc_auc': roc_auc_score(
            y_true=y_true,
            y_score=y_pred_prob,
            average='weighted'
        ),
        'avg_prec': average_precision_score(
            y_true=y_true,
            y_score=y_pred_prob,
            average='weighted'
        ),
        'f1': f1_score(
            y_true=y_true,
            y_pred=y_pred_label,
            average='binary'
        ),
        'auprc': au_prc(
            y_true=y_true,
            y_pred_prob=y_pred_prob
        )}
    return methods.get(method, 'Invalid method')


def au_prc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Computes the area under the precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Array of {0,1} ground truth labels.
    y_pred_prob : np.ndarray
        Array of [0,1] predicted probabilities.

    Returns
    -------
    _auc : float
        Area under the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred_prob)
    return auc(x=recall, y=precision)


def remove_feature_duplication(list_of_arrays: list) -> set:
    """
    Removes duplication of features in a list.

    Parameters
    ----------
    list_of_arrays : list
        List of arrays.

    Returns
    -------
    set
        Set of unique features.
    """

    return set(np.concatenate(list_of_arrays)) if list_of_arrays else set(list_of_arrays)


if __name__ == "__main__":
    print(create_balanced_distributions.__doc__)
