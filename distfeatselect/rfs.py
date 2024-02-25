from sklearn.linear_model import LogisticRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

import numpy as np
import pandas as pd
from dcor import distance_correlation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import statsmodels.discrete.discrete_model as sm
import warnings
from utils import model_score
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import _safe_indexing

SVM_classifier = SVC(kernel='linear')

class RFS(BaseEstimator, TransformerMixin):
    """
    Transformer that performs Randomized Feature Selection (RFS).

    Read more in the official documentation of the Python package.

    Parameters
    ----------
    n_models : int
        Number of models generated per iteration. Default=100.
    n_iters : int
        Number of iterations. Default is 300.
    tuning : float
        Learning rate that dictates the speed of regressor inclusion probability (rip) convergence.
        Smaller values -> slower convergence. Default is 10.
    tol : float
        Tolerance condition. Default is 0.002.
    alpha : float
        Significance level for model pruning. Default is 0.99.
    rip_cutoff : float
        Determines rip threshold for feature inclusion in final model. Default=1.
    metric : str
        Optimization metric. Default='roc_auc'. Options: 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'.
    mu_init : dict
        Dictionary containing user-assigned RIPs. If None, RIPs are initialised to 1/n_features.
    method : str
        RFSC method. Options (str): 'dcor', 'logit', 'L1_logit'. Refer to the package documentation for more details.
    estimator : estimator instance, optional
        An unfitted estimator. Used for evaluation of the model with selected features.
        If method 'dcor', estimator is not used. Default is SVC(kernel='linear')
    verbose : bool
        Provides extra information. Default is False.
    """

    def __init__(self,
            n_models: int=100,
            n_iters: int=300,
            tuning: float=10,
            tol: float=0.002,
            alpha: float=0.99,
            rip_cutoff: float=1,
            metric: str='roc_auc',
            mu_init: dict=None,
            method = "logit",
            estimator = SVM_classifier,
            verbose: bool=False,
        ):
        self.n_models = n_models
        self.n_iters = n_iters
        self.tol = tol
        self.alpha = alpha
        self.tuning = tuning
        self.rip_cutoff = rip_cutoff
        self.metric = metric
        self.mu_init = mu_init
        self.method = method
        self.estimator = estimator
        self.verbose = verbose

        if self.metric not in ['acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc']:
            raise ValueError(f"metric must be one of 'acc', 'roc_auc', 'weighted', 'avg_prec', 'f1', 'auprc'. Received: {self.metric} ")
        if self.method not in ['logit', 'L1_logit', 'dcor']:
            raise ValueError(f"method must be one of 'logit', 'L1_logit', 'dcor'. Received: {self.method}")
        if not isinstance(self.n_models, int) and self.n_models <= 0:
            raise TypeError(f"n_models parameter must be an integer greater than 0. Received: {type(self.n_models)}")
        if not isinstance(self.n_iters, int) and self.n_iters <= 0:
            raise TypeError(f"n_iters parameter must be an integer greater than 0. Received: {type(self.n_iters)}")
        if self.tol < 0:
            raise ValueError(f"tol parameter must be a positive number. Received: {self.tol}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha parameter must be between 0 and 1. Received: {self.alpha}")
        if self.tuning < 0:
            raise ValueError(f"tuning parameter must be a positive number. Received: {self.tuning}")
        if not 0 < self.rip_cutoff <= 1:
            raise ValueError(f"rip_cutoff parameter must be between 0 and 1. Received: {self.rip_cutoff}")
        if not hasattr(self.estimator, 'fit'):
            raise ValueError("estimator must be an estimator instance with a fit method.")


        print(
            f"{self.__class__.__name__} Initialised with with parameters: \n \
            n_models = {n_models}, \n \
            n_iters = {n_iters}, \n \
            method = {method}, \n \
            estimator = {estimator}, \n \
            mu_init = {mu_init}, \n \
            tuning = {tuning}, \n \
            tol = {tol}, \n \
            rip_cutoff = {rip_cutoff}, \n \
            metric = {metric}, \n \
            alpha = {alpha} \n ------------"
        ) if self.verbose else None

    def __repr__(self) -> str:
        """
        Returns a string representation of the object. It includes the class name and the values of the instance variables.

        Returns
        -------
        str
            The string representation of the object.
        """
        return (f"{self.__class__.__name__}(n_models={self.n_models}, n_iters={self.n_iters}, \n \
               method = {self.method}, estimator = {self.estimator}, tuning={self.tuning}, \n \
               metric={self.metric}, alpha={self.alpha}), mu_init = {self.mu_init},  tol = {self.tol}, \n \
               rip_cutoff = {self.rip_cutoff}")


    def fit(self, X, y):
        """
        Learn the features to select from X.

        This is the main part of RFS algorithm. It extracts the model populations and evaluates
        them on the validation set, and updates the feature inclusion probabilities accordingly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        perf_break = False
        self.perf_check = 0
        self.rnd_feats = {}
        self.sig_feats = {}
        avg_model_size = np.empty((0,))
        avg_performance = np.empty((0,))

        _, self.n_features  = np.shape(X)
        if self.mu_init is None:
          self.mu_init = (1/self.n_features) * np.ones((self.n_features))

        mu = self.mu_init

        X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


        for t in range(self.n_iters):
            mask, performance_vector, size_vector = self.generate_models(
                                                            X_train=X_train,
                                                            Y_train=Y_train,
                                                            X_val=X_val,
                                                            Y_val=Y_val,
                                                            mu=mu
                                                        )
            mu_update = self.update_feature_probability(
                            mask=mask,
                            performance=performance_vector,
                            mu=mu
                        )
            avg_model_size = np.append(avg_model_size, np.mean(size_vector.ravel()[np.flatnonzero(performance_vector)]))
            avg_performance = np.append(avg_performance, np.mean(performance_vector.ravel()[np.flatnonzero(performance_vector)]))

            if perf_check(t, avg_performance, self.tol):
                self.perf_check += 1
            else:
                self.perf_check = 0

            print(f"iter: {t}, avg model size: {avg_model_size[t]:.2f}, avg perf is: {avg_performance[t]:.5f}, tol not reached, max diff is: {np.abs(mu_update - mu).max():.5f}, perf check: {self.perf_check}.") if self.verbose else None

            if tol_check(mu_update, mu, self.tol): # stop if tolerance is reached.
                print(f"Tol reached. Number of features above rip_cutoff is {np.count_nonzero(mu_update>=self.rip_cutoff)}")
                break

            elif self.perf_check >= 2:
                perf_break = True
                break

            mu = mu_update

        self.iters = t
        if perf_break is True:
            sorted_items = sorted(self.sig_feats.items(), key=lambda item: item[1], reverse=True)
            self.features_ = sorted_items[0][0]  # Get the key (features) with the highest performance

        else:
            self.features_ = select_model(mu=mu, rip_cutoff=self.rip_cutoff)
            #case no feature meets the threshold
            if len(self.features_) == 0:
                self.features_ = list(np.nonzero(mu)[0])

        #If none of the features are selected, then choose n_feats features with the highest mu_init values
        if len(self.features_) == 0:
            n_feats = self.n_features if self.n_features < 10 else int(self.n_features*0.3)

            indices_and_values = sorted(enumerate(self.mu_init), key=lambda x: x[1], reverse=True)
            max_indices = [index for index, value in indices_and_values[:n_feats]]
            self.features_ = max_indices

        if self.method == "dcor":
            self.best_performance_ = distance_correlation(X_train[:, self.features_], Y_train)
        if self.method in ["logit", "L1_logit"]:
            model = sm.Logit(Y_train, X_train[:, self.features_]).fit_regularized(method='l1', alpha=0.1)
            prediction = model.predict(X_val[:, self.features_])

            self.best_performance_ = model_score(
                method=self.metric,
                y_true=Y_val,
                y_pred_label=prediction.round(),
                y_pred_prob=prediction
            )
        return self

    def update_feature_probability(
        self,
        mask: np.ndarray,
        performance: np.ndarray,
        mu: np.ndarray
    ) -> np.ndarray:
        """
        Updates the feature probability vector mu based on the performance of the models generated.

        Parameters
        ----------
        mask : np.ndarray
            Matrix of shape (n_models, n_features) containing the mask of the models generated.
        performance : np.ndarray
            Performance evaluation for each model.
        mu : np.ndarray
            Current feature probability vector.

        Returns
        -------
        mu_update : np.ndarray
            Updated feature probability vector.
        """
        features_incld = np.sum(mask, axis=0) #(n_features,)
        features_excld = (np.ones(len(mu)) * self.n_models) - features_incld #(n_features,)
        features_performance = performance @ mask #(n_features,)

        ## evaluate importance of features
        with np.errstate(divide='ignore', invalid='ignore'):
            E_J_incld = features_performance / features_incld
            E_J_excld = (np.sum(performance) - features_performance) / features_excld

        # for where features not chosen in any models
        E_J_incld[np.isnan(E_J_incld)] = 0
        E_J_excld[np.isnan(E_J_excld)] = 0
        E_J_excld[np.isinf(E_J_excld)] = 0

        gamma = gamma_update(performance=performance, tuning=self.tuning)
        _mu = mu + gamma*(E_J_incld - E_J_excld)
        return np.clip(_mu, 0, 1)

    def generate_models(
            self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            X_val: np.ndarray,
            Y_val: np.ndarray,
            mu: np.ndarray
        ):
        """
        Generates random models and for each model evaluates the significance of each feature.
        Statistically significant features are retained and resultant model's performance on validation partition is
        evaluated and stored.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        Y_train : np.ndarray
            Training labels.
        X_val : np.ndarray
            Validation data.
        Y_val : np.ndarray
            Validation labels.
        mu : np.ndarray
            Array of regressor inclusion probabilities of each feature.

        Returns
        -------
        mask_mtx : np.ndarray
            Matrix containing 1 in row i at column j if feature j was included in model i, else 0.
        performance_vector : np.ndarray
            Array containing performance of each model.
        size_vector : np.ndarray
            Array containing number of features in each model.
        """

        if isinstance(X_train, pd.DataFrame):
          X_train = X_train.to_numpy()
        if isinstance(Y_train, pd.DataFrame):
          Y_train = Y_train.to_numpy()
        if isinstance(X_val, pd.DataFrame):
          X_val = X_val.to_numpy()
        if isinstance(Y_val, pd.DataFrame):
          Y_val = Y_val.to_numpy()

        mask = np.empty((0,))
        mask_mtx = np.zeros((len(mu),)) # mask matrix
        performance_vector = np.zeros((self.n_models,))# performance vector
        size_vector = np.zeros((self.n_models,)) # average model size vector
        #mu[0] = 1 # set bias term to 1

        for i in range(self.n_models):
            count = 0
            mask_vector = np.zeros((len(mu),))
            while True:
                generated_features = generate_model(mu)
                if len(generated_features) < 1:
                    old_mu_0 = mu[0]
                    mu[0] = 1
                    generated_features = generate_model(mu)
                    mu[0] = old_mu_0

                if tuple(generated_features) not in self.rnd_feats.keys(): # check if model has been generated before
                    if self.method == "logit":
                      logreg_init = sm.Logit(
                                        Y_train,
                                        X_train[:, generated_features]
                                    ).fit(disp=False, method='lbfgs')
                      #Statistical Test for Regressor Significance
                      #The rejection of redundant terms is a crucial step in the identification procedure.
                      significant_features = prune_model(
                                                model=logreg_init,
                                                feature_ids=generated_features,
                                                alpha=self.alpha
                                            )

                    elif self.method == "L1_logit":
                      lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
                      lr.fit(X_train[:, generated_features], Y_train)
                      significant_features = np.where(lr.coef_[0] != 0)[0]

                    elif self.method == "dcor":
                      significant_features = generated_features

                    self.rnd_feats[tuple(generated_features)] = significant_features

                else: # if model has been generated before, use the stored significant features
                    significant_features = self.rnd_feats[tuple(generated_features)]

                if len(significant_features) > 1:
                    break

                count += 1
                if count > 1000:
                    self.alpha -= 0.05
                    warnings.warn("The significance level alpha was reduced by 0.05")


            size_vector[i] = len(significant_features)
            mask_vector[significant_features] = 1
            mask = np.concatenate((mask, mask_vector), axis=0)

            if tuple(significant_features) not in self.sig_feats.keys(): # check if model has been evaluated before
                if self.method in ["L1_logit", "logit"]:
                    model = self.estimator.fit(X_train[:, significant_features], Y_train)
                    prediction = model.predict(X_val[:, significant_features])

                    performance_vector[i] = model_score(
                                              method=self.metric,
                                              y_true=Y_val,
                                              y_pred_label=prediction.round(),
                                              y_pred_prob=prediction
                                          )
                elif self.method == "dcor":
                    performance_vector[i] = distance_correlation(X_train[:, significant_features], Y_train)

                self.sig_feats[tuple(significant_features)] = performance_vector[i]

            else: # if model has been evaluated before, used the stored performance
                performance_vector[i] = self.sig_feats[tuple(significant_features)]

        mask_mtx = np.reshape(mask, (len(performance_vector), len(mu)))
        return mask_mtx, performance_vector, size_vector

    def get_support(self, indices = False):
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
        mask[list(self.features_)] = True
        return self.features_ if indices else mask

    def get_best_performance(self):
        """
        Retrieve the best performance achieved.
        This method returns the performance of the model with selected features.

        Returns
        -------
        float
            The best performance value if available, otherwise 0.
        """
        return self.best_performance_ if self.best_performance_ is not None else 0


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


def select_model(mu: np.ndarray, rip_cutoff: float) -> list:
    """
    Selects final model based on features that are above the regressor inclusion probability (rip) threshold.

    Parameters
    ----------
    mu : np.ndarray
        Current feature probability vector.
    rip_cutoff : float
        Regressor inclusion probability threshold.

    Returns
    -------
    list
        List of features that are above the rip threshold.
    """
    return list((mu >= rip_cutoff).nonzero()[0])

def tol_check(mu_update: np.ndarray, mu: np.ndarray, tol: float):
    """
    Checks if maximum difference between mu vectors is below tolerance threshold.

    Parameters
    ----------
    mu_update : np.ndarray
        Mu at the iteration t+1.
    mu : np.ndarray
        Mu at the iteration t.
    tol : float
        Tolerance condition.

    Returns
    -------
    bool
        True max difference below tolerance, else False.
    """
    return np.abs(mu_update - mu).max() < tol

def perf_check(iter: int, avg_perf: np.ndarray, tol: float) -> bool:
    """
    Checks if performance has converged based on tolerance threshold.

    Parameters
    ----------
    iter : int
        Current iteration.
    avg_perf : np.ndarray
        Average performance vector.
    tol : float
        Tolerance condition.

    Returns
    -------
    bool
        True if performance converged
        (difference between average performance of two consecutive iterations is less than or equal to tol), else False.
    """
    return iter > 2 and np.abs(avg_perf[iter] - avg_perf[iter-1]) <= tol

def gamma_update(
        performance: np.ndarray,
        tuning: float=10
    ) -> float:
    """
    Scale the update of the feature probability vector.

    Parameters
    ----------
    performance : np.ndarray
        Performance evaluation for each model.
    tuning : float, optional
        Tuning parameter to adjust convergence rate, default=10.

    Returns
    -------
    gamma : float
        Scaling factor for the update of the feature probability vector.
    """
    return 1/(tuning*(np.max(performance) - np.mean(performance)) + 0.1)

def prune_model(
        model: object,
        feature_ids: list,
        alpha: float
    ) -> list:
    """
    Tests whether features are significant at selected significance level. Returns index of significant features.

    Parameters
    ----------
    model : object
        Logistic regression model object. See statsmodels.api.Logit.
    feature_ids : list
        Feature ids included in the model.
    alpha : float
        (0,1) significance level.

    Returns
    -------
    list
        List of features above the significance level.
    """
    return list(set(feature_ids[np.where(model.pvalues<=alpha)]))

def generate_model(mu: np.ndarray) -> np.ndarray:
    """
    Takes a vector of probabilities and returns a random model.

    Parameters
    ----------
    mu : np.ndarray
        Array of probabilities for each feature.

    Returns
    -------
    index : np.ndarray
        Randomly generated numbers corresponding to features ids based on probabilities.
        Array of selected features - their indices.

    Raises
    ------
    ValueError
        Array of probabilities has all zero probabilities.
    """
    if np.count_nonzero(mu) == 0:
        raise ValueError("mu cannot be all zeros")

    index= [0]
    while len(index) <= 1:
        index = np.flatnonzero(np.random.binomial(1,mu))
    return index