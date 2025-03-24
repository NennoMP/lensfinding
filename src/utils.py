"""Module containing helpers for plotting."""
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# Matplotlib settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


##############################
# Helpers to visualize class distributions
##############################
def plot_class_distributions(
    dataloaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    dataloader_names: Tuple[str, str],
    class_names: List[str] = None,
) -> None:
    """Plot class distributions for two PyTorch dataloaders.
    
    Args:
        dataloaders: tuple of two dataloaders to visualize.
        dataloader_names: tuple of names for each dataloader (e.g., 'Train', 'Test').
        class_names: list of class names to be used in the plot.
    """
    num_dataloaders = len(dataloaders)
    assert num_dataloaders == 2, 'Exactly two dataloaders are expected.'
    assert len(dataloader_names) == num_dataloaders, 'Number of dataloader names must match number of dataloaders.'
    
    fig, axes = plt.subplots(1, num_dataloaders, figsize=(10, 4))
    
    for i, (dataloader, name, ax) in enumerate(zip(dataloaders, dataloader_names, axes)):
        label_counts = _get_label_counts(dataloader)
        total_samples = sum(label_counts.values())
        
        if class_names is not None:
            x_labels = [class_names[i] if i < len(class_names) else str(i) for i in label_counts.keys()]
        else:
            x_labels = [str(i) for i in label_counts.keys()]
        
        bars = ax.bar(x_labels, label_counts.values(), color='steelblue', width=0.2)
        percentages = [count/total_samples * 100 for count in label_counts.values()]
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + (total_samples * 0.005),
                f'({percentages[j]:.1f}%)',
                ha='center',
                va='bottom'
            )
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Titles and labels
        ax.set_title(f'{name} set', pad=15)
        ax.set_xlabel('Classes')
        if i == 0:
            ax.set_ylabel('Frequency')

    fig.suptitle('Class Distribution', y=0.9, x=0.55)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def _get_label_counts(dataloader, target_idx=1) -> Dict[int, int]:
    """Helper function to extract and count labels from a dataloader.
    
    Args:
        target_idx: index of the target in the dataloader output.
        
    Returns:
        A dictionary mapping class indices to counts.
    """
    # Collect all labels from the dataloader
    all_labels = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            labels = batch[target_idx]
        else:
            try:
                labels = batch['labels']
            except (KeyError, TypeError):
                raise ValueError(
                    "Couldn't extract labels from dataloader. Please specify the correct 'target_idx'."
                )
        
        # Convert possible tensor labels to numpy for counting
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        # Flatten in case of batched labels
        if hasattr(labels, 'flatten'):
            labels = labels.flatten()
        
        all_labels.extend(labels)

    return dict(sorted(Counter(all_labels).items()))

##############################
# Helpers to visualize predictions
##############################
def plot_predictions(labels: np.ndarray, predictions: np.ndarray, class_names: List[str]) -> None:
    """Plot ROC curve and confusion matrix.
    Args:
        labels: true labels.
        predictions: model predictions.
        class_names: list of class names to be used in the plot.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    labels = np.array(labels).ravel()
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Check if multi-class or binary.class
    multi_class = n_classes > 2
    is_pred_2d = len(predictions.shape) > 1 and predictions.shape[1] >= 1
    
    # Plot ROC curve
    if multi_class and is_pred_2d and predictions.shape[1] > 2: # multi-class
        y_bin = label_binarize(labels, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        for i in range(n_classes):
            ax1.plot(fpr[i], tpr[i], lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    else: # binary-class
        if n_classes == 2:
            pos_label = 1
        else:
            pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            
        if is_pred_2d:
            if predictions.shape[1] == 2:
                pred_scores = predictions[:, 1]
            else:
                pred_scores = predictions.ravel()
        else:
            pred_scores = predictions
            
        fpr, tpr, _ = roc_curve(labels, pred_scores, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    # Plot 1 - Spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    if is_pred_2d and predictions.shape[1] > 1:
        y_pred = np.argmax(predictions, axis=1)
    else:
        if np.all(np.logical_or(predictions == 0, predictions == 1)) or len(np.unique(predictions)) <= 2:
            y_pred = predictions.ravel().astype(int)
        else:
            # If predictions are scores, threshold at 0.5
            y_pred = (predictions.ravel() > 0.5).astype(int)
    
    # Plot confusion matrix
    cm = confusion_matrix(labels, y_pred)
    
    # Calculate percentages for confusion matrix
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)  # Avoid division by zero
    cm_perc = cm / cm_sum * 100
    
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    tick_marks = np.arange(min(n_classes, len(class_names)))
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(class_names[:len(tick_marks)])
    ax2.set_yticklabels(class_names[:len(tick_marks)])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Display percentage with one decimal point
            ax2.text(j, i, f"{cm_perc[i, j]:.1f}%",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax2.set_ylabel('True label')
    ax2.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.show()