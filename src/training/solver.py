"""Module implementing training and inference logic for training a PyTorch model."""
import os
import copy
from typing import Optional, Tuple

import torch
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score

class Solver:
    """Class implementing training logic to train and test a PyTorch model.

    The Solver class is responsible for training and testing a PyTorch model. The model will be 
    trained with a learning rate decay approach. The class also supports early stopping based on
    the validation performance.

    Attributes:
        scheduler: learning rate scheduler.
        best_params: best model weights based on validation performance.
        best_val_auc: a float indicating the highest validation AUC achieved.
        patience_counter: an integer indicating the number of epochs since the last improvement.
        history: a dictionary containing training/inference history.
    """
    def __init__(
        self, 
        device: torch.device, 
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: Optional[torch.utils.data.DataLoader] = None, 
        criterion = torch.nn.CrossEntropyLoss(),
        lr: float = 0.001,
        lr_decay: float = 1e-5,
        weight_decay: float = 0,
    ) -> None:
        """
        Args:
            device: device to run the model on.
            model: PyTorch model to be trained.
            train_dataloader: dataloader for training data.
            val_dataloader: dataloader for inference data.
            lr: learning rate.
            weight_decay: L2 regularization strength.
            lr_decay: amount to decrease the learning rate at each epoch.
        """
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        
        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0, (lr - epoch * lr_decay) / lr) # linear decay
        )
    
        self._reset()

    def _reset(self) -> None:
        """Reset solver state."""
        self.best_val_auc = float('-inf')
        self.best_params = None
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': [],
            'learning_rate': []
        }

    def run_epoch(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        training: bool = True
    ) -> Tuple[float, float]:
        """Perform one epoch of training or inference.
        
        Args: 
            training: specifies whether model should be in training or inference mode.
        """
        self.model.train() if training else self.model.eval()

        running_loss = 0.0
        all_labels, all_preds = [], []
        with torch.set_grad_enabled(training):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.append(
                    torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()
                )
        
        # Compute metrics on current epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        all_labels = np.array(all_labels)
        all_preds = np.vstack(all_preds)
        epoch_auc = roc_auc_score(
            all_labels, 
            all_preds[:, 1] if all_preds.shape[1] == 2 else all_preds,
            multi_class='ovr' if all_preds.shape[1] > 2 else 'raise', 
            average='macro' if all_preds.shape[1] > 2 else None
        )

        return (epoch_loss, epoch_auc)

    def train(self, epochs: int = 10, patience: Optional[int] = None, log: bool = False) -> None:
        """Train the model with early stopping and learning rate decay.
        
        Args:
            epochs: number of training epochs.
            patience: number of epochs to wait for improvement before early stopping.
            log: whether to log metrics to wandb.
        """
        patience = patience if patience is not None else epochs
        
        for epoch in range(epochs):
            print('-' * 10)
            print(f'Epoch {epoch+1}/{epochs}')

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.8f}')

            train_loss, train_auc = self.run_epoch(self.train_dataloader) # training
            val_loss, val_auc = self.run_epoch(self.val_dataloader, training=False) # inference
            self.scheduler.step()

            # Print and store current epoch metrics
            print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)

            # Log to wandb
            if log:
                wandb.log({
                    'train_loss': train_loss,
                    'train_auc': train_auc,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                })  

            # Check if validation performance improved
            self.patience_counter += 1
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0

            # Check for early-stopping
            if self.patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} - Best Validation AUC: {self.best_val_auc:.4f}')
                break