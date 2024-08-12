import numpy as np
from typing import List, Union

def _check_inputs(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> (np.ndarray, np.ndarray):
    """Helper function to check the validity of input arrays."""
    if not isinstance(y_true, (list, np.ndarray)):
        raise TypeError("y_true should be a list or numpy array.")
    if not isinstance(y_pred, (list, np.ndarray)):
        raise TypeError("y_pred should be a list or numpy array.")
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional.")
    return y_true, y_pred

def accuracy(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the accuracy score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    
    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
        Accuracy classification score.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    return round(np.mean(y_true == y_pred), 6)

def precision(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the precision score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
        Precision classification score.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return round(tp / (tp + fp), 6)

def recall(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the recall score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
        Recall classification score.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return round(tp / (tp + fn), 6)

def f_score(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray], beta: float = 1.0) -> float:
    """
    Compute the F-score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    beta : float, default=1.0
        Weight of precision in harmonic mean.

    Returns
    -------
    score : float
        F-score.
    """
    if beta <= 0:
        raise ValueError("beta should be > 0 in the F-score calculation")
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return round((1 + beta**2) * (p * r) / (beta**2 * p + r), 6)

def roc_auc(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the ROC AUC score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
        ROC AUC score.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        raise ValueError("roc_auc score is not defined in cases with no positive or no negative samples.")
    return round(np.sum((y_pred[y_true == 1].reshape(-1, 1) > y_pred[y_true == 0].reshape(1, -1)).astype(int)) / (pos * neg), 6)

def mae(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        Mean Absolute Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    return round(np.mean(np.abs(y_true - y_pred)), 6)

def mse(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        Mean Squared Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    return round(np.mean((y_true - y_pred)**2), 6)

def rmse(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        Root Mean Squared Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 6)

def mape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        Mean Absolute Percentage Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    epsilon = np.finfo(np.float64).eps
    return round(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100, 6)

def wape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Weighted Absolute Percentage Error (WAPE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        Weighted Absolute Percentage Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    epsilon = np.finfo(np.float64).eps
    return round((np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + epsilon)) * 100, 6)

def smape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        Symmetric Mean Absolute Percentage Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    epsilon = np.finfo(np.float64).eps
    return round(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100, 6)

def mase(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray], y_train: Union[List[float], np.ndarray]) -> float:
    """
    Compute the Mean Absolute Scaled Error (MASE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : array-like of shape (n_samples,)
        Predicted values.

    y_train : array-like of shape (n_samples,)
        Training data values to compute scale.

    Returns
    -------
    score : float
        Mean Absolute Scaled Error.
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    y_train = np.asarray(y_train)
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1-dimensional.")
    n = len(y_train)
    if n < 2:
        raise ValueError("y_train must contain at least two elements.")
    d = np.sum(np.abs(y_train[1:] - y_train[:-1])) / (n - 1)
    if d == 0:
        raise ValueError("The training data y_train has zero variance.")
    errors = np.abs(y_true - y_pred)
    return round(np.mean(errors / d), 6)
