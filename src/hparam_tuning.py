"""Module implementing hyperparameter tuning logic for model selection."""
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import wandb
os.environ['WANDB_SILENT'] = 'true'
from torch.utils.data import DataLoader

from src.training.solver import Solver


def create_run_name(config: Dict[str, Any]) -> str:
    """Create a run name based on the configuration provided.
    
    Args:
        config: a dictionary containing the hyperparameters' configuration.
    """
    parts = []
    for key, value in config.items():
        parts.append(f"{key}-{value}")
    result = "_".join(parts)
    return result


class Tuner:
    """Class implementing hyperparameter tuning logic for model selection.

    The Tuner class is reponsible for performing model selection on a PyTorch model, leveraging 
    wandb sweeps. Different configurations will be explored based on the specified strategy (i.e., 
    grid-search or random-search), and their results logged to wandb dashboard.
    """
    def __init__(
        self, 
        device: torch.device,
        project_name: str,
        project_entity: str,
        model_fn: Callable[[Dict[str, Any]], nn.Module],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        configs: Union[Dict, List],
        criterion = torch.nn.CrossEntropyLoss(),
        epochs: int = 100,
        patience: Optional[int] = None,
        save_weights: bool = False,
    ) -> None:
        """_summary_

        Args:
            project_name: the name of the wandb project to log the results to.
            project_entity: the wandb entity to which the project belongs.
            model_fn: a function that returns an instance of the model to be trained.
            configs: a dictionary containing the hyperparameters, and corresponding intervals, to 
            be explored.
            criterion: the loss function to be used for training.
            save_weights: whether to save the best model weights after training.
        """
        self.device = device
        self.model_fn = model_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.configs = configs
        self.criterion = criterion

        # Project
        self.project_name = project_name
        self.project_entity = project_entity
        
        self.epochs = epochs
        self.patience = patience
        self.save_weights = save_weights

        self.reset_()

    def reset_(self) -> None:
        """Reset tuner state."""
        self.config_count = 0
        self.n_configs = 1

    def run(self, method: str = 'grid', n_search: int = None, sweep_id = None) -> None:
        """Perform hyperparameter tuning leveraging wandb sweeps.
        
        Args:
            method: the search strategy to be used. Options are 'grid' or 'random'.
            n_search: the number of configurations to explore. Only used for random search.
            sweep_id: the wandb sweep id to be used. If None, a new sweep will be created.
        """
        # grid-search
        if method == 'grid':
            for params in self.configs.values():
                self.n_configs *= len(params['values'])
        # random-search
        elif method == 'random':
            if n_search:
                self.n_configs = n_search
            else:
                self.n_configs = '---'
        else:
            raise ValueError(f"Invalid tuning method: {method}. Options are 'grid' and 'random'.")
        
        self.n_search = n_search
        if sweep_id is None:
            sweep_config = {'method': method, 'parameters': self.configs}
            sweep_id = wandb.sweep(sweep_config, entity=self.project_entity, project=self.project_name)
        wandb.agent(
            sweep_id, 
            entity=self.project_entity, 
            function=self.test_config_, 
            project=self.project_name, 
            count=n_search
        )
    
    def test_config_(self, config=None) -> None:
        """Test a given configuration.
        
        Args:
            config: the configuration to be tested.
        """
        with wandb.init(config=config) as run:
            config = wandb.config
            run.name = create_run_name(config)
            
            print(f"\nEvaluating Config #{self.config_count + 1} [of {self.n_configs}]: {config}")

            # Create and train the model with the given configuration
            model = self.model_fn()
            solver = Solver(
                device=self.device,
                model=model,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                criterion=self.criterion,
                lr=config.lr,
                lr_decay=config.lr_decay,
            )
            solver.train(epochs=self.epochs, patience=self.patience, log=True)

            # Save model weights
            if self.save_weights:
                stretch = 'stretch' if config.apply_sqrt_stretch else 'nostretch'
                solver.save_model_weights(
                    f'weights/{self.project_name}/{stretch}/', 
                    f'weights_{config.trial_id}.pth'
                )

        self.config_count += 1