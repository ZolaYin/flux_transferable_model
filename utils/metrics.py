# utils/metrics.py
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np


def calculate_metrics(all_targets, all_predictions, average='weighted'):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average=average,
        zero_division=0
    )

    accuracy = accuracy_score(all_targets, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }


def calculate_per_class_accuracy(all_targets, all_predictions, num_classes: int):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    cm = confusion_matrix(all_targets, all_predictions, labels=np.arange(num_classes))

    per_class_acc = np.zeros(num_classes, dtype=float)
    for i in range(num_classes):
        tp = cm[i, i]
        class_total_samples = np.sum(cm[i, :])
        if class_total_samples > 0:
            per_class_acc[i] = tp / class_total_samples
        else:
            per_class_acc[i] = 0.0

    return per_class_acc


def calculate_regression_metrics(all_targets, all_predictions):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    all_targets = np.asarray(all_targets, dtype=np.float64)
    all_predictions = np.asarray(all_predictions, dtype=np.float64)

    flat_targets = all_targets.reshape(-1)
    flat_predictions = all_predictions.reshape(-1)

    mse = mean_squared_error(flat_targets, flat_predictions)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(flat_targets, flat_predictions))
    r2 = float(r2_score(flat_targets, flat_predictions))

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': float(mse),
    }

    if all_targets.ndim == 2 and all_targets.shape[1] > 1:
        for idx in range(all_targets.shape[1]):
            target_i = all_targets[:, idx]
            pred_i = all_predictions[:, idx]
            mse_i = mean_squared_error(target_i, pred_i)
            metrics[f'rmse_{idx}'] = float(np.sqrt(mse_i))
            metrics[f'mae_{idx}'] = float(mean_absolute_error(target_i, pred_i))
            metrics[f'r2_{idx}'] = float(r2_score(target_i, pred_i))
            metrics[f'mse_{idx}'] = float(mse_i)

    return metrics


def calculate_site_regression_metrics(all_targets, all_predictions, site_ids):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    all_targets = np.asarray(all_targets, dtype=np.float64)
    all_predictions = np.asarray(all_predictions, dtype=np.float64)
    site_ids = np.asarray(site_ids)

    if all_targets.ndim > 1 and all_targets.shape[1] > 1:
        # CarbonBench reports each flux separately; this helper summarizes the
        # flattened output only when a caller passes multi-task predictions.
        all_targets = all_targets.reshape(len(site_ids), -1)
        all_predictions = all_predictions.reshape(len(site_ids), -1)
    else:
        all_targets = all_targets.reshape(len(site_ids), -1)
        all_predictions = all_predictions.reshape(len(site_ids), -1)

    site_rmse, site_mae, site_r2 = [], [], []
    for site in np.unique(site_ids):
        mask = site_ids == site
        if int(mask.sum()) < 2:
            continue
        target_i = all_targets[mask].reshape(-1)
        pred_i = all_predictions[mask].reshape(-1)
        if target_i.size < 2:
            continue
        mse_i = mean_squared_error(target_i, pred_i)
        site_rmse.append(float(np.sqrt(mse_i)))
        site_mae.append(float(mean_absolute_error(target_i, pred_i)))
        site_r2.append(float(r2_score(target_i, pred_i)))

    metrics = {'site_n': int(len(site_r2))}
    for name, values in [('rmse', site_rmse), ('mae', site_mae), ('r2', site_r2)]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            metrics[f'site_{name}_p25'] = float('nan')
            metrics[f'site_{name}_median'] = float('nan')
            metrics[f'site_{name}_p75'] = float('nan')
            continue
        metrics[f'site_{name}_p25'] = float(np.quantile(arr, 0.25))
        metrics[f'site_{name}_median'] = float(np.quantile(arr, 0.50))
        metrics[f'site_{name}_p75'] = float(np.quantile(arr, 0.75))
    return metrics


def calculate_target_site_regression_metrics(all_targets, all_predictions, site_ids, target_names=None):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    all_targets = np.asarray(all_targets, dtype=np.float64)
    all_predictions = np.asarray(all_predictions, dtype=np.float64)
    site_ids = np.asarray(site_ids)
    if all_targets.ndim == 1:
        all_targets = all_targets[:, None]
        all_predictions = all_predictions[:, None]

    target_names = target_names or [str(i) for i in range(all_targets.shape[1])]
    metrics = {}
    for idx, name in enumerate(target_names):
        site_rmse, site_mae, site_r2 = [], [], []
        for site in np.unique(site_ids):
            mask = site_ids == site
            if int(mask.sum()) < 2:
                continue
            target_i = all_targets[mask, idx]
            pred_i = all_predictions[mask, idx]
            finite = np.isfinite(target_i) & np.isfinite(pred_i)
            if int(finite.sum()) < 2:
                continue
            target_i = target_i[finite]
            pred_i = pred_i[finite]
            mse_i = mean_squared_error(target_i, pred_i)
            site_rmse.append(float(np.sqrt(mse_i)))
            site_mae.append(float(mean_absolute_error(target_i, pred_i)))
            site_r2.append(float(r2_score(target_i, pred_i)))

        safe_name = name.replace('/', '_').replace(' ', '_')
        metrics[f'site_n_{safe_name}'] = int(len(site_r2))
        for metric_name, values in [('rmse', site_rmse), ('mae', site_mae), ('r2', site_r2)]:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                metrics[f'site_{metric_name}_p25_{safe_name}'] = float('nan')
                metrics[f'site_{metric_name}_median_{safe_name}'] = float('nan')
                metrics[f'site_{metric_name}_p75_{safe_name}'] = float('nan')
                continue
            metrics[f'site_{metric_name}_p25_{safe_name}'] = float(np.quantile(arr, 0.25))
            metrics[f'site_{metric_name}_median_{safe_name}'] = float(np.quantile(arr, 0.50))
            metrics[f'site_{metric_name}_p75_{safe_name}'] = float(np.quantile(arr, 0.75))
    return metrics
