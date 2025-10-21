import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple

predicted = np.array([
    1,1,1,0,1,0,1,1,0,0
])
actual = np.array([
    1,1,1,1,0,0,1,0,0,0
])

big_size = 500000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)


@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel() # din matrice de dim n transforma in array
    return tp, fp, fn, tn


@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))

    return tp, fp, fn, tn

assert tp_fp_fn_tn_sklearn(actual, predicted) == tp_fp_fn_tn_numpy(actual, predicted)



@timed(use_seconds=True, show_args=True)
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return accuracy_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
   tp,fp,fn,tn = tp_fp_fn_tn_numpy(gt, pred)
   return (tp + tn) / (tp + tn + fp + fn)


assert accuracy_sklearn(actual, predicted) == accuracy_numpy(actual, predicted)
rez_1 = accuracy_sklearn(big_actual, big_predicted)
rez_2 = accuracy_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)


@timed(use_seconds=True, show_args=True)
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return f1_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    tp,fp,fn,tn = tp_fp_fn_tn_numpy(gt,pred)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# assert f1_score_sklearn(actual, predicted) == f1_score_numpy(actual, predicted)
assert np.isclose(rez_1, rez_2)
rez_1 = f1_score_sklearn(big_actual, big_predicted)
rez_2 = f1_score_numpy(big_actual, big_predicted)

print(f"rez1 = {rez_1}")
print(f"rez2 = {rez_2}")
