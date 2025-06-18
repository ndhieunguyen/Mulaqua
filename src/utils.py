import evaluate
import numpy as np
from torch.nn.functional import softmax
from torch import tensor
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix


clf_metrics = evaluate.combine(
    ["accuracy", "f1", "precision", "recall", "matthews_correlation"]
)
auc = evaluate.load("roc_auc")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions[0], axis=1)
    prediction_scores = softmax(tensor(predictions[0]), dim=-1)
    prediction_scores = prediction_scores[:, 1].cpu().numpy()

    metrics = clf_metrics.compute(predictions=preds, references=labels)
    metrics.update(auc.compute(prediction_scores=prediction_scores, references=labels))
    metrics.update(
        {
            "eval_balanced_acc": balanced_accuracy_score(y_pred=preds, y_true=labels),
            "eval_gmean": geometric_mean_score(y_pred=preds, y_true=labels),
        }
    )
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    metrics.update(
        {
            "eval_specificity": specificity,
            "eval_tn": tn,
            "eval_fp": fp,
            "eval_fn": fn,
            "eval_tp": tp,
        }
    )
    metrics = dict(sorted(metrics.items()))
    return metrics
