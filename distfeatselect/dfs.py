import functools
import itertools
import multiprocessing as mp
import time
import warnings
from typing import Union, Tuple

import numpy as np
import pandas as pd

import statsmodels.discrete.discrete_model as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted

from .utils import create_balanced_distributions
from .utils import evaluate_interim_model
from .utils import remove_feature_duplication


class DFS(BaseEstimator, TransformerMixin):
    """
    Transformer that performs Distributed Feature Selection (DFS).

    Read more in the official documentation of the Python package.

    Parameters
    ----------
    local_fs_method : object, optional
            The machine learning model used for feature selection for each data partition. Default is a Support Vector
            Classifier (SVC) with a linear kernel and a random seed of 42.
    n_vbins : int, optional
            Number of vertical partitions to create for the data. Defaults to 1.
    n_hbins : int, optional
            Number of horizontal partitions to create for the data. If output = 'ensemble', each hbin will converge to
            its own best model. Defaults to 1.
    n_runs : int, optional
            Number of feature-sharing iterations to perform. Larger numbers may yield better results, but also take longer.
            Defaults to 1.
    redistribute_features : bool, optional
            If True, the base features included in each bin will be shuffled at each feature-sharing iteration.
            Does not affect feature sharing. Defaults to False.
    feature_sharing : str, optional
            The method used to share features. Defaults to 'all'. Options (str): 'all', 'latest', 'top_k'.
            If feature_sharing = 'all', the entire history of best features from all sub-processes will be shared.
            If feature_sharing = 'latest', features from all sub-processes at the current iteration will be shared.
            If feature_sharing = 'top_k', the best k features will be shared.
    k : int, optional
            Number of best features to share. Only used if feature_sharing = 'top_k'. Defaults to 0.
    output : str, optional
            Output type desired. Options (str): 'single', 'ensemble'. If output = 'single', the best model from all
            sub-processes will be returned. If output = 'ensemble', the best model from each horizontal partition will be returned.
            If output = 'ensemble', no features between different horizontal partitions will be created. Defaults to 'single'.
    metric : str, optional
            Evaluation metric used in the optimization process.
            Options (str) : ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']. Defaults to 'roc_auc'.
            For more information on the metrics, see the documentation for the sklearn.metrics module.
    verbose : bool, optional
            If True, prints extra information. Defaults to False.
    max_processes : int, optional
            Enforces maximum number of processes that can be generated. If None, will use all available cores.
            Defaults to None.
    estimator : estimator instance, optional
            An unfitted estimator. Machine learning model used for evaluation of the results for all sub-processes.
    """

    def __init__(
            self,
            local_fs_method=SelectKBest(score_func=chi2, k=3),
            n_vbins: int = 1,
            n_hbins: int = 1,
            n_runs: int = 1,
            redistribute_features: bool = False,
            feature_sharing: str = 'all',
            k: int = 0,
            output: str = 'single',
            metric: str = 'roc_auc',
            verbose: bool = False,
            max_processes: int = None,
            estimator=None
    ):
        # validation
        if local_fs_method is not None and not hasattr(local_fs_method, 'fit'):
            raise ValueError("local_fs_method must be an instance with a fit method.")

        if not isinstance(n_vbins, int) or n_vbins < 1:
            raise ValueError("n_vbins must be a positive integer.")

        if not isinstance(n_hbins, int) or n_hbins < 1:
            raise ValueError("n_hbins must be a positive integer.")

        if not isinstance(n_runs, int) or n_runs < 1:
            raise ValueError("n_runs must be a positive integer.")

        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer.")

        if max_processes is not None and (not isinstance(max_processes, int) or max_processes < 1):
            raise ValueError("max_processes must be None or a positive integer.")

        if not isinstance(redistribute_features, bool):
            raise ValueError("redistribute_features must be a boolean value.")

        if feature_sharing not in ['all', 'latest', 'top_k']:
            raise ValueError("feature_sharing must be one of 'all', 'latest', or 'top_k'.")

        if not isinstance(output, str) or output not in ['single', 'ensemble']:
            raise ValueError("output must be either 'single' or 'ensemble'.")

        if metric not in ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']:
            raise ValueError("metric must be one of 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', or 'auprc'.")

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value.")

        if estimator is not None and not hasattr(estimator, 'fit'):
            raise ValueError("estimator must be an estimator instance with a fit method.")

        self.estimator = estimator
        self.metric = metric
        self.n_vbins = n_vbins
        self.n_hbins = n_hbins
        self.n_runs = n_runs
        self.redistribute_features = redistribute_features
        self.feature_sharing = feature_sharing
        self.k = k
        self.output = output
        self.verbose = verbose
        self.loaded_data = 0
        self.max_processes = max(max_processes, mp.cpu_count()) if max_processes is not None else mp.cpu_count()
        self.local_fs_method = local_fs_method

        print(f"{self.__class__.__name__} Initialised with parameters: \n \
        local_fs_method = {local_fs_method}, \n \
        n_vbins = {n_vbins}, \n \
        n_hbins = {n_hbins}, \n \
        n_runs = {n_runs}, \n \
        redistribute = {redistribute_features}, \n \
        sharing = {feature_sharing}, \n \
        k = {k}, \n \
        output = {output}, \n \
        metric = {metric}, \n \
        estimator = {estimator}, \n \
        max_processes is {self.max_processes} \n ------------") if self.verbose else None

    def __repr__(self) -> str:
        """
        Returns a string representation of the object. It includes the class name and the values of the instance variables.

        Returns
        -------
        str
            The string representation of the object.

        """
        return f"{self.__class__.__name__}(local_fs_method = {self.local_fs_method}, n_vbins={self.n_vbins}, \
                 n_hbins={self.n_hbins}, n_runs={self.n_runs}, redistribute_features={self.redistribute_features}, \
                 feature_sharing={self.feature_sharing}, k={self.k}, output={self.output}, \
                 metric={self.metric}, verbose={self.verbose}, max_processes={self.max_processes}, estimator={self.estimator})"

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], Y_train: Union[np.ndarray, pd.DataFrame]):
        """
        Learn the features to select from X_train.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features/predictors.

        Y_train : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()

        if isinstance(Y_train, (pd.DataFrame, pd.Series)):
            Y_train = Y_train.to_numpy()

        self.n_samples, self.n_features = X_train.shape

        # create vertical and horizontal partitions
        distributed_features, distributed_samples = create_balanced_distributions(
            labels=Y_train,
            n_feats=self.n_features,
            n_vbins=self.n_vbins,
            n_hbins=self.n_hbins
        )

        # Initialization
        self.J_star = {i: [] for i in range(self.n_hbins)}
        self.J_best = {i: [0, 0] for i in range(self.n_hbins)}
        self.results_full = {}
        self.M_history = {}
        self.n_iter_conv = self.n_runs
        M = {i: 0 for i in range(self.n_hbins)}  # selected features
        non_converged_hbins = np.arange(self.n_hbins).tolist()

        if self.verbose:
            print(
                f"Number of Samples: {self.n_samples}. Horizontal Disitribution SHAPE: {np.shape(distributed_samples)}")
            print(
                f"Number of Features: {self.n_features}. Vertical Distribution SHAPE: {np.shape(distributed_features)}")

        if self.n_hbins == 1 and self.n_vbins == 1:
            if self.output == 'ensemble':
                print(
                    "WARNING: Ensemble output is not possible with n_hbins = 1 and n_vbins = 1. Setting output = 'single'")
            self.output = 'single'

            self.local_fs_method.fit(X_train, Y_train)
            self.local_fs_method.get_support()
            self.J_best = {0: [self.local_fs_method.get_support()]}
            self.build_final_model(J_best=self.J_best)
            return self.J_best

        self.time_per_iteration = []
        for r in range(self.n_runs):
            start_time = time.time()
            iter_results = {}  # initialise dictionary for storing results

            if self.redistribute_features:
                distributed_features, _ = create_balanced_distributions(
                    labels=Y_train,
                    n_feats=self.n_features,
                    n_vbins=self.n_vbins,
                    n_hbins=self.n_hbins
                )
            result_obj = []

            def store_results(obj, indices, features_passed):  # callback for mp
                performance = obj.get_best_performance() if hasattr(obj, 'get_best_performance') else None
                result = self._Result(list(obj.get_support(indices=True)), indices, features_passed,
                                      evaluation=performance)
                result_obj.append(result)

            pool = mp.Pool(processes=min((self.n_vbins * len(non_converged_hbins)), self.max_processes),
                           maxtasksperchild=1)
            for i, j in itertools.product(range(self.n_vbins), non_converged_hbins):
                feature_partition = list(distributed_features[:, i])
                feature_share = self.join_features(
                    features=feature_partition,
                    M=M[j]
                )
                features_passed = [int(i) for i in feature_share]
                sample_indices = list(distributed_samples[:, j])

                pool.apply_async(
                    self.local_fs_method.fit,
                    args=(X_train[:, features_passed][sample_indices, :], Y_train[sample_indices]),
                    callback=functools.partial(store_results, indices=(r, i, j), features_passed=features_passed)
                )

            pool.close()
            pool.join()

            if len(result_obj) != (
                    self.n_vbins * len(non_converged_hbins)):
                print(
                    f"result_obj length is {len(result_obj)}. Should be {(self.n_vbins * len(non_converged_hbins))}")

            for result in result_obj:
                # predict on all sub-processes
                global_features = [result.features_passed[i] for i in result.features]
                if result.evaluation is None:
                    result.model, result.evaluation = evaluate_interim_model(
                        model_features=global_features,
                        X=X_train,
                        y=Y_train,
                        metric=self.metric,
                        model=self.estimator
                    )

                iter_results[result.drfsc_index] = result

            # check if every result is here
            for i, j in itertools.product(range(self.n_vbins), non_converged_hbins):
                if (r, i, j) not in [x.drfsc_index for x in result_obj]:
                    print(f"missing result {(r, i, j)}")
                    iter_results[(r, i, j)] = [[0], 0, self.output, [0]]

            # update full results dict
            self.results_full, single_iter_results = self.update_full_results(
                results_full=self.results_full,
                iter_results=iter_results
            )

            # map local feature indices to global feature indices
            single_iter_results = self.map_local_feats_to_gt(
                iter_results=single_iter_results,
                r=r,
                hbins=non_converged_hbins
            )

            comb_sig_feats_gt = [model[0] for model in single_iter_results.values()]

            # update the current best results
            self.J_best, self.J_star = _update_best_models(
                J_best=self.J_best,
                J_star=self.J_star,
                single_iter_results=single_iter_results,
                non_converged_hbins=non_converged_hbins,
                metric=self.metric
            )

            # update converged horizontal partitions
            non_converged_hbins = self.convergence_check(
                r=r,
                J_star=self.J_star,
                non_converged_hbins=non_converged_hbins
            )

            # update feature list shared with other partitions
            M = self.feature_share(
                r=r,
                results_full=self.results_full,
                comb_sig_feats_gt=comb_sig_feats_gt,
                non_converged_hbins=non_converged_hbins,
                M=M
            )

            print(f"M: {M}") if self.verbose else None
            self.M_history.update([(r, M)])

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.time_per_iteration.append(elapsed_time)
            if len(non_converged_hbins) == 0:
                self.n_iter_conv = r + 1
                print(f"All horizontal partitions have converged. Final iter count: {r + 1}")
                break

        self.labels = Y_train
        self.data = X_train
        self.build_final_model(J_best=self.J_best)

        for value in self.results_full.values():
            # remove the features_passed from results_full
            value.pop()

        return self

    def get_support(self, indices=False):
        """
        Get the feature support mask or indices.

        Parameters
        ----------
        indices : bool, optional
            If True, returns the indices of the supported features.
            If False (default), returns a boolean mask indicating supported features.

        Returns
        -------
        numpy.ndarray
            If indices is True, a numpy array containing the indices of the supported features.
            If indices is False, a boolean mask indicating supported features.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted.
        """
        check_is_fitted(self)
        mask = np.zeros(self.n_features, dtype=bool)

        if type(self.features_) is int:
            mask[self.features_] = True
        else:
            mask[list(self.features_)] = True
        return self.features_ if indices else mask

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        mask = self.get_support()
        if not mask.any():
            warnings.warn(
                (
                    "No features were selected: either the data is"
                    " too noisy or the selection test too strict."
                ),
                UserWarning,
            )
            if hasattr(X, "iloc"):
                return X.iloc[:, :0]
            return np.empty(0, dtype=X.dtype).reshape((X.shape[0], 0))
        return _safe_indexing(X, mask, axis=1)

    def feature_share(
            self,
            r: int,
            results_full: dict,
            comb_sig_feats_gt: list,
            non_converged_hbins: list,
            M: dict
    ):
        """
        Computes the features to be shared with each bin in the subsequent iteration.

        Parameters
        ----------
        r : int
            Current iteration.
        results_full : dict
            Dictionary containing the results from all iterations.
        comb_sig_feats_gt : list
            List of global feature indices from models in the current iteration.
        non_converged_hbins : list
            List of horizontal partition indices that have not converged.

        Returns
        -------
        M : dict
            Dictionary containing the features to be shared with each bin in the subsequent iteration.
        """
        if self.feature_sharing == 'latest':
            M = {i: 0 for i in range(self.n_hbins)}  # reset M dict if feature sharing is set to latest

        for j in non_converged_hbins:
            if self.output == 'ensemble':
                M[j] = remove_feature_duplication(
                    [results_full[(r, i, j)][0] for i in range(self.n_vbins)])

            elif self.feature_sharing == 'top_k':
                top_k_model_feats = [sorted(results_full.values(), key=lambda x: x[1], reverse=True)[i][0] for i in
                                     range(min(self.k,
                                               len(results_full.values())))]
                M[j] = remove_feature_duplication(top_k_model_feats)

            else:
                M[j] = remove_feature_duplication(
                    comb_sig_feats_gt)

        return M

    def final_model(self, model_ensemble: dict) -> None:
        """
        Helper function for generating the final model based on the ensemble of models.

        Parameters
        ----------
        model_ensemble : dict
            Contains the ensemble of models. Each key is a separate model, and the value is a list containing a list of
            the feature indices used for that model and the model object itself.
        """

        if self.output != 'ensemble':
            raise ValueError("Final model only valid for ensemble output")

        idx = range(self.data.shape[1])

        df = pd.DataFrame(columns=model_ensemble.keys(), index=idx)
        for key, value in model_ensemble.items():
            coefs = value[1].params
            feat_index = value[0]
            for val in zip(feat_index, coefs):
                df.loc[val[0], key] = val[1]

        df.fillna(0, inplace=True)
        df['mean'] = df.mean(axis=1)

        self.model_coef = np.array(df[df['mean'] != 0]['mean'])  # mean for every feature coeff
        self.model_features_num = list(df[df['mean'] != 0].index)

    def convergence_check(
            self,
            r: int,
            J_star: dict,
            non_converged_hbins: list
    ) -> list:
        """
        Checks if the tolerance condition has been met for the current iteration.

        Parameters
        ----------
        r : int
            Current iteration number.
        J_star : dict
            Dictionary of best models from each horizontal partition.
        non_converged_hbins : list
            List of horizontal partitions that have not converged.

        Returns
        -------
        hbins_not_converged : list
            Indices of horizontal partition that have not converged.
        """
        hbins_converged = []
        for hbin in non_converged_hbins:
            if J_star[hbin] == 1:
                print(f"Iter {r}. The best model in hbin {hbin} cannot be improved further") if self.verbose else None
                hbins_converged.append(hbin)

            elif r >= 2 and J_star[hbin][r] == J_star[hbin][r - 1] and J_star[hbin][r] == J_star[hbin][r - 2]:
                print(
                    f"Iter {r}. No appreciable improvement over the last 3 iterations in hbin {hbin}") if self.verbose else None
                hbins_converged.append(hbin)

        non_converged_set = set(non_converged_hbins)
        converged_set = set(hbins_converged)
        return list(non_converged_set - converged_set)

    def map_local_feats_to_gt(
            self,
            iter_results: dict,
            r: int,
            hbins: list
    ) -> dict:
        """
        Maps local feature indices to global feature indices for each model in the current iteration.

        Parameters
        ----------
        iter_results : dict
            Dictionary with the results of the iteration.
        r : int
            Number of the current iteration.
        hbins : list
            List of horizontal partitions that have not converged.

        Returns
        -------
        iter_results : dict
            Dict updated with global feature indices.
        """
        for i, j in itertools.product(range(self.n_vbins), hbins):
            iter_results[(r, i, j)][0] = list(np.array(iter_results[(r, i, j)][2])[list(iter_results[(r, i, j)][0])])

        return iter_results

    def join_features(self, features: list, M: Union[set, int]) -> list:
        """
        Joins the feature partitions to the relevant information from previous iterations.

        Parameters
        ----------
        features : list
            Feature partition (list of features for the partition).
        M : set or int
            Features selected at the previous round, feature list shared with other partitions.

        Returns
        -------
        list
            List of feature partition augmented with M (features selected at the previous round).
        """
        if isinstance(M, int):
            return list(set(features).union([M]))

        if isinstance(M, set):
            return list(set(features).union(M))

    def update_full_results(
            self,
            results_full: dict,
            iter_results: dict
    ):
        """
        Updates the full results dictionary with the results from the current iteration.

        Parameters
        ----------
        results_full : dict
            Dictionary containing the results from all iterations.
        iter_results : dict
            Dictionary containing the results from the current iteration.

        Returns
        -------
        results_full : dict
            Updated full result dictionary.
        single_iter_results : dict
            Dictionary containing the results from the current iteration where values are list of 3 elements:
            selected features, evaluation and initial features (one partition augmented with features from previous round)
            on which feature selection is applied.
        """
        single_iter_results = {
            result.drfsc_index:
                [result.features, result.evaluation, result.features_passed] for result in iter_results.values()
        }
        results_full |= single_iter_results
        return results_full, single_iter_results

    def build_final_model(self, J_best):
        """
        Builds the final model based on the output specified.
        Output options: 'single', 'ensemble'.

        Parameters
        ----------
        J_best : dict
            Dictionary containing the best model for each horizontal partition.
        """

        if self.output == 'ensemble':
            ensemble = {}
            for h_bin in range(self.n_hbins):
                model = sm.Logit(
                    self.labels,
                    self.data[:, J_best[h_bin][0]]
                ).fit(disp=False, method='lbfgs')
                ensemble[f"model_h{str(h_bin)}"] = [J_best[h_bin][0], model]

            self.ensemble = ensemble
            self.final_model(self.ensemble)
            self.features_num = self.model_features_num

        else:
            self.features_num = _select_single_model(J_best=J_best)[0]

        self.features_ = self.features_num
        self.model = sm.Logit(
            self.labels,
            self.data[:, self.features_num]
        ).fit_regularized(method='l1', alpha=0.1)

        self.coef_ = self.model.params

    class _Result():
        def __init__(self, features: list, drfsc_index, features_passed, evaluation=None, model=None):
            self.evaluation = evaluation
            self.features = features
            self.features_passed = features_passed
            self.drfsc_index = drfsc_index
            self.model = model


def _select_single_model(J_best: dict) -> list:
    """
    Returns model with the highest performance evaluation.

    Parameters
    ----------
    J_best : dict
        Dictionary containing as keys the horizontal partition index and as values the best model for that partition (list)
        in terms of feature indices, performance evaluation (float), and metric used for evaluation (str).

    Returns
    -------
    best_model : list
        List containing the best model for the entire dataset.
    """
    return sorted(J_best.values(), key=lambda x: x[1], reverse=True)[0]

def _update_best_models(
    J_best: dict,
    J_star: dict,
    single_iter_results: dict,
    non_converged_hbins: list,
    metric: str
) -> Tuple[dict, dict]:
    """
            Compares results from the current iteration against current best models.
            If a model from the current iteration is better, it is saved.

            Parameters
            ----------
            J_best : dict
                Dictionary containing as keys the horizontal partition index and as values the best model for that partition.
            J_star : dict
                Dictionary containing only the performance evaluation of the best model for each horizontal partition (list).

            Returns
            -------
            J_best : dict
                Dictionary containing as keys the horizontal partition index and as values the best model for that partition (list)
                in terms of feature indices, performance evaluation (float), and metric used for evaluation (str).
            J_star : dict
                Dictionary containing only the performance evaluation of the best model for each horizontal partition (list).
            """
    for key, model in single_iter_results.items():
        if key[2] in non_converged_hbins and model[1] > J_best[key[2]][1]:
            print(f"New best model for hbin {key[2]}. {metric}={round(model[1], 5)} -- Model features {model[0]}")
            J_best[key[2]] = [model[i] for i in range(3)]
    for j in non_converged_hbins:
        J_star[j].append(J_best[j][1])

    return J_best, J_star