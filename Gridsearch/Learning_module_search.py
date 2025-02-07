import os
import glob
import json
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# ------------------------------
# Data Loading Functions
# ------------------------------
def load_mat_file(mat_file):
    """
    Attempts to load features and labels from a .mat file by checking several candidate keys.
    Returns (X, y, keys) if found, otherwise (None, None, list_of_keys).
    """
    data = scipy.io.loadmat(mat_file)
    candidate_X_keys = ['X', 'x', 'data', 'expression', 'expr']
    candidate_y_keys = ['y', 'Y', 'labels', 'class', 'target']
    
    X = None
    y = None
    for key in candidate_X_keys:
        if key in data:
            X = data[key]
            break
    for key in candidate_y_keys:
        if key in data:
            y = data[key]
            break
    return X, y, list(data.keys())

def load_bio_datasets(data_dir):
    """
    Loads bio data from all .mat files in the given directory.
    For each file, it searches for the feature matrix and label vector using common keys.
    Returns a list of tuples: (filename, X, y)
    """
    mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
    print(f"Found {len(mat_files)} .mat files in '{data_dir}'.")
    
    datasets = []
    for mat_file in mat_files:
        X, y, keys = load_mat_file(mat_file)
        if X is not None and y is not None:
            print(f"File '{mat_file}': X shape {X.shape}, y shape {np.array(y).shape}")
            y = np.squeeze(y)  # ensure y is 1D
            datasets.append((mat_file, X, y))
        else:
            print(f"Warning: File '{mat_file}' does not contain expected keys. Found keys: {keys}")
    return datasets

# ------------------------------
# Standalone Learning Module (Classifier Only)
# ------------------------------
class NonLinearLayer(nn.Module):
    """
    A configurable non-linear activation layer that supports different activation functions.
    """
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
        self.activation_functions = {
            'tanh': torch.tanh,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'sigmoid': torch.sigmoid,
            'selu': F.selu,
            'softplus': F.softplus
        }
        if activation_type not in self.activation_functions:
            raise ValueError(f"Unsupported activation: {activation_type}")
            
    def forward(self, x):
        return self.activation_functions[self.activation_type](x)

class StandaloneLearningModule(nn.Module):
    """
    A flexible learning module (classifier) that supports an arbitrary number of hidden layers
    and configurable activation functions. This module does not include the attention mechanism.
    """
    def __init__(self, input_size, hidden_layers=[500], num_classes=10, 
                 activation='relu', dropout_rate=0.0):
        super().__init__()
        layers = []
        in_features = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(NonLinearLayer(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ------------------------------
# Training and Validation Functions
# ------------------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    epoch_loss = running_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    epoch_loss = running_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc

# ------------------------------
# Grid Search with k-Fold Cross Validation
# ------------------------------
def search_best_learning_module_cv(X, y, input_size, num_classes, device,
                                   num_folds=5, num_epochs=40):
    """
    Performs a grid search over candidate configurations for the standalone learning module
    using stratified k-fold cross validation.
    
    Candidate hyperparameters include:
      - Hidden layer architectures
      - Activation functions
      - Learning rates
      - Dropout rates
      
    Returns a list of the top 3 configurations (by average validation accuracy) along with their details.
    """
    # Candidate hyperparameters
    candidate_hidden_layers = [
        [128],
        [256],
        [512],
        [1024],
        [256, 128],
        [512, 256],
        [512, 512],
        [256, 128, 64],
        [512, 256, 128]
    ]
    candidate_activations = ['relu', 'tanh', 'leaky_relu', 'elu', 'sigmoid', 'selu']
    candidate_learning_rates = [0.01, 0.001, 0.005, 0.0005]
    candidate_dropout_rates = [0.0, 0.1, 0.2]
    
    # Build candidate configurations (Cartesian product)
    candidate_configs = []
    for hidden in candidate_hidden_layers:
        for act in candidate_activations:
            for lr in candidate_learning_rates:
                for dr in candidate_dropout_rates:
                    candidate_configs.append({
                        "hidden_layers": hidden,
                        "activation": act,
                        "learning_rate": lr,
                        "dropout_rate": dr
                    })
    
    # Stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = []  # Will hold tuples: (config, avg_val_acc)
    
    for config in candidate_configs:
        fold_val_accs = []
        print("Testing configuration:", config)
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            
            # Create tensors and DataLoaders
            X_train_tensor = torch.tensor(X_train_cv, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_cv, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val_cv, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_cv, dtype=torch.long)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Create model with current configuration
            model = StandaloneLearningModule(
                input_size=input_size,
                hidden_layers=config["hidden_layers"],
                num_classes=num_classes,
                activation=config["activation"],
                dropout_rate=config["dropout_rate"]
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
            criterion = nn.CrossEntropyLoss()
            
            # Train for the specified number of epochs
            for epoch in range(num_epochs):
                train(model, train_loader, criterion, optimizer, device)
            # Validate after training
            _, val_acc = validate(model, val_loader, criterion, device)
            fold_val_accs.append(val_acc)
            print(f"  Fold {fold+1}/{num_folds} | Val Acc: {val_acc:.4f}")
        
        avg_val_acc = np.mean(fold_val_accs)
        results.append((config, avg_val_acc))
        print(f"Configuration {config} average validation accuracy: {avg_val_acc:.4f}")
        print("----------------------------------------------------")
    
    # Sort and extract top 3 configurations
    results.sort(key=lambda x: x[1], reverse=True)
    top_3 = results[:20]
    
    print("\n=== Top 3 Configurations ===")
    for idx, (cfg, acc) in enumerate(top_3, start=1):
        print(f"Rank {idx}: {cfg} with average validation accuracy: {acc:.4f}")
    
    return top_3

# ------------------------------
# Main Routine: Process Each Dataset and Save Results
# ------------------------------
if __name__ == "__main__":
    data_dir = "data_bio"  # Directory containing your .mat files
    datasets = load_bio_datasets(data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_results = []  # To store best configs for each dataset
    
    # Process each dataset separately
    for filename, X_all, y_all in datasets:
        print("\n========================================")
        print(f"Processing dataset from file: {filename}")
        print("Dataset summary:")
        print(" - Total samples:", X_all.shape[0])
        print(" - Number of features:", X_all.shape[1])
        unique_labels = np.unique(y_all)
        print(" - Unique labels (before remapping):", unique_labels)
        
        # Remap labels if necessary (e.g., if labels start at 1)
        if np.min(unique_labels) > 0:
            print("Remapping labels to start at 0.")
            y_all = y_all - np.min(unique_labels)
        print(" - Unique labels (after remapping):", np.unique(y_all))
        
        # Use the full dataset for cross validation
        print("Performing stratified 5-fold cross validation search...")
        top_configs = search_best_learning_module_cv(
            X=np.array(X_all), y=np.array(y_all),
            input_size=X_all.shape[1],
            num_classes=len(np.unique(y_all)),
            device=device,
            num_folds=5,
            num_epochs=40  # Increase epochs to allow better convergence
        )
        
        # Record results for the current dataset
        dataset_result = {
            "filename": filename,
            "total_samples": int(X_all.shape[0]),
            "num_features": int(X_all.shape[1]),
            "unique_labels": np.unique(y_all).tolist(),
            "top_configs": [{"config": cfg, "avg_val_acc": acc} for cfg, acc in top_configs]
        }
        final_results.append(dataset_result)
        print("========================================")
    
    # Save final results to a JSON file
    output_file = "best_configs.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nFinal best configurations for all datasets have been saved to '{output_file}'.")