import copy
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def create_model(one_output = False):
  if one_output:
    outdimen = 1
  else:
    outdimen = 2
  model = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, outdimen)
        )

  return model

def train_model(model, train_loader, test_loader, n_epochs=1500, learning_rate=7e-5, weight_decay=1e-5, patience=50, factor=0.9):
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    
    best_mse = np.inf
    best_model_weights = None
    history = {'train_loss': [], 'test_loss': [], 'lr': [], 'train_density_mse': [], 'train_B4_mse': [], 'test_density_mse': [], 'test_B4_mse': []}
    
    for epoch in range(n_epochs):
        model.train()
        train_losses, train_density_losses, train_B4_losses = [], [], []
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            
            # Compute total loss
            loss = loss_fn(y_pred, y_batch)

            # Compute individual MSEs
            density_mse = loss_fn(y_pred[:, 0], y_batch[:, 0])
            B4_mse = loss_fn(y_pred[:, 1], y_batch[:, 1])

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_density_losses.append(density_mse.item())
            train_B4_losses.append(B4_mse.item())
        
        train_loss = np.mean(train_losses)
        train_density_mse = np.mean(train_density_losses)
        train_B4_mse = np.mean(train_B4_losses)
        
        history['train_loss'].append(train_loss)
        history['train_density_mse'].append(train_density_mse)
        history['train_B4_mse'].append(train_B4_mse)
        
        model.eval()
        test_losses, test_density_losses, test_B4_losses = [], [], []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                
                # Compute total loss
                loss = loss_fn(y_pred, y_batch)

                # Compute individual MSEs
                density_mse = loss_fn(y_pred[:, 0], y_batch[:, 0])
                B4_mse = loss_fn(y_pred[:, 1], y_batch[:, 1])

                test_losses.append(loss.item())
                test_density_losses.append(density_mse.item())
                test_B4_losses.append(B4_mse.item())
        
        test_loss = np.mean(test_losses)
        test_density_mse = np.mean(test_density_losses)
        test_B4_mse = np.mean(test_B4_losses)
        
        history['test_loss'].append(test_loss)
        history['test_density_mse'].append(test_density_mse)
        history['test_B4_mse'].append(test_B4_mse)

        # Step the scheduler
        scheduler.step(test_loss)
        
        # Record the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        if (epoch+1) % 50 == 0 or epoch == n_epochs - 1:
            print(f"Epoch: {epoch}, Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}, LR: {current_lr:.2e}")
            print(f"   Train Density MSE: {train_density_mse:.4f}, Train B4 MSE: {train_B4_mse:.4f}")
            print(f"   Test Density MSE: {test_density_mse:.4f}, Test B4 MSE: {test_B4_mse:.4f}")
        
        if test_loss < best_mse:
            best_mse = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
    
    return best_model_weights, best_mse, history

class VerboseCallback:
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, result):
        # Print status every 'interval' calls
        if len(result.x_iters) % self.interval == 0:
            print(f"Call {len(result.x_iters)}: Current Best MSE = {result.fun}")

