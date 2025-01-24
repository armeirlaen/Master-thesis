import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import pandas as pd
from sklearn.model_selection import KFold
import json
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from modellinear import AFS
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AFSArchitectureSearch:
    def __init__(self, train_data, train_labels, val_data, val_labels, 
                 test_data, test_labels, device=None):
        self.train_data = torch.FloatTensor(train_data)
        self.train_labels = torch.LongTensor(train_labels)
        self.val_data = torch.FloatTensor(val_data)
        self.val_labels = torch.LongTensor(val_labels)
        self.test_data = torch.FloatTensor(test_data)
        self.test_labels = torch.LongTensor(test_labels)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
    def get_architecture_configs(self):
        """Define different architectural configurations to test"""
        return {
            # Attention Module Params
            'e_node': [16, 32, 64, 128],  # Extraction network size
            'a_node': [2, 4, 6, 8],  # Attention network output size
            
            # Non-linearity configurations
            'extract_activation': ['tanh', 'relu', 'gelu', 'elu'],  # Extraction network activation
            'attention_activation': ['tanh', 'sigmoid', 'relu'],  # Attention network activation
            'hidden_activation': ['relu', 'leaky_relu', 'elu'],  # Learning module activation
            
            # Learning Module Params
            'hidden_size': [128, 256],  # Size of hidden layer
            'dropout_rate': [0.0, 0.2],  # Dropout rate for learning module
            
            # Training Params
            'batch_size': [32],
            'learning_rate': [0.1],
            'weight_decay': [0.0001],
            'num_epochs': [80]
        }
        
    def train_model(self, model, train_loader, val_loader, config):
        """Train a single model configuration"""
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(config['num_epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            attention_weights = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs, attention = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()
                    attention_weights.append(attention.cpu())
                    print("Loss:",val_loss)
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 5:
                    break
        
        # Restore best model
        model.load_state_dict(best_model_state)
        return model, val_accuracy, val_loss

    def evaluate_model(self, model, data_loader):
        """Evaluate model performance with additional metrics"""
        model.eval()
        correct = 0
        total = 0
        attention_weights = []
        
        # For computing precision, recall, and F1 per class
        num_classes = len(torch.unique(self.train_labels))
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs, attention = model(batch_x)
                _, predicted = outputs.max(1)
                
                # Update confusion matrix
                for t, p in zip(batch_y.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                attention_weights.append(attention.cpu())
        
        # Calculate accuracy
        accuracy = correct / total
        
        # Calculate per-class precision, recall, and F1
        precision = torch.zeros(num_classes)
        recall = torch.zeros(num_classes)
        f1 = torch.zeros(num_classes)
        
        for i in range(num_classes):
            # Precision
            precision[i] = confusion_matrix[i, i] / confusion_matrix[:, i].sum() if confusion_matrix[:, i].sum() != 0 else 0
            # Recall
            recall[i] = confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() != 0 else 0
            # F1
            precision_recall_sum = precision[i] + recall[i]
            f1[i] = 2 * precision[i] * recall[i] / precision_recall_sum if precision_recall_sum > 0 else 0
        
        # Calculate macro-averaged metrics
        macro_precision = precision.mean().item()
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()
        
        # Calculate attention statistics
        attention_tensor = torch.cat(attention_weights)
        attention_mean = attention_tensor.mean(dim=0)
        attention_std = attention_tensor.std(dim=0)
        
        # Calculate sparsity metrics
        # 1. L1 sparsity (normalized by feature count)
        l1_sparsity = attention_mean.abs().mean().item()
        
        # 2. Feature selection ratio (features with attention > threshold)
        threshold = 0.5  # Can be adjusted
        selected_features = (attention_mean > threshold).float().mean().item()
        
        # 3. Gini coefficient for attention distribution
        sorted_attention = torch.sort(attention_mean)[0]
        n = len(sorted_attention)
        index = torch.arange(1, n + 1, device=sorted_attention.device)
        gini = ((2 * index - n - 1) * sorted_attention).sum() / (n * sorted_attention.sum())
        gini = gini.item()
        
        return {
            'accuracy': accuracy,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1,
            'per_class_precision': precision.numpy(),
            'per_class_recall': recall.numpy(),
            'per_class_f1': f1.numpy(),
            'attention_mean': attention_mean.numpy(),
            'attention_std': attention_std.numpy(),
            'sparsity_l1': l1_sparsity,
            'feature_selection_ratio': selected_features,
            'gini_coefficient': gini
        }

    def search(self):
        """Perform architecture search"""
        configs = self.get_architecture_configs()
        keys, values = zip(*configs.items())
        
        for v in product(*values):
            config = dict(zip(keys, v))
            print(f"\nTesting configuration: {config}")
            
            # Verify labels
            num_classes = len(torch.unique(self.train_labels))
            min_label = torch.min(self.train_labels).item()
            max_label = torch.max(self.train_labels).item()
            
            #print(f"\nLabel verification:")
            #print(f"Number of classes: {num_classes}")
            #print(f"Label range: {min_label} to {max_label}")
            
            assert min_label == 0, "Labels must start from 0"
            assert max_label == num_classes - 1, "Labels must be consecutive"
            
            # Create data loaders
            train_dataset = TensorDataset(self.train_data, self.train_labels)
            val_dataset = TensorDataset(self.val_data, self.val_labels)
            test_dataset = TensorDataset(self.test_data, self.test_labels)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False
            )
            
            # Create and train model
            model = AFS(
                input_size=self.train_data.shape[1],
                num_classes=num_classes,
                e_node=config['e_node'],
                a_node=config['a_node'],
                hidden_size=config['hidden_size'],
                extract_activation=config['extract_activation'],
                attention_activation=config['attention_activation'],
                hidden_activation=config['hidden_activation'],
                dropout_rate=config['dropout_rate']
            ).to(self.device)
           
            model, val_accuracy, val_loss = self.train_model(
                model, train_loader, val_loader, config
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, test_loader)
            
            # Store results
            result = {
            'config': config,
            'metrics': {
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1_score'],
                'sparsity_l1': test_metrics['sparsity_l1'],
                'feature_selection_ratio': test_metrics['feature_selection_ratio'],
                'gini_coefficient': test_metrics['gini_coefficient']
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()

    def save_results(self, filename='afs_architecture_search_results.json'):
        """Save search results"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def analyze_results(self):
        """Analyze search results"""
        df = pd.DataFrame([
            {**r['config'], **r['metrics']}
            for r in self.results
        ])
        
        # Best configurations
        best_configs = {
            metric: df.loc[df[metric].idxmax()].to_dict()
            for metric in ['val_accuracy', 'test_accuracy']
        }
        
        # Parameter impact analysis
        param_importance = {}
        for param in self.get_architecture_configs().keys():
            correlation = df[param].corr(df['test_accuracy'])
            param_importance[param] = abs(correlation)
        
        return {
            'best_configs': best_configs,
            'param_importance': param_importance,
            'full_results': df
        }

def run_architecture_search(train_data, train_labels, val_data, val_labels, 
                          test_data, test_labels):
    """Run complete architecture search process"""
    search = AFSArchitectureSearch(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels
    )
    
    print("Starting architecture search...")
    search.search()
    
    print("\nAnalyzing results...")
    analysis = search.analyze_results()
    
    print("\nBest Configurations:")
    for metric, config in analysis['best_configs'].items():
        print(f"\nBest for {metric}:")
        for param, value in config.items():
            if param not in ['val_accuracy', 'test_accuracy', 'val_loss']:
                print(f"  {param}: {value}")
    
    print("\nParameter Importance:")
    for param, importance in sorted(
        analysis['param_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{param}: {importance:.4f}")
    
    return search, analysis

def get_data_from_mat(data):
    """Extract features and labels from mat file with different possible key names"""
    # Common key pairs for features and labels
    key_pairs = [
        ('data', 'label'),  # Standard names
        ('X', 'Y'),         # Alternative names
        ('x', 'y'),         # Lower case
        ('data', 'target'), # scikit-learn style
        ('features', 'labels')  # Another common format
    ]
    
    # Try to find matching pair
    for feat_key, label_key in key_pairs:
        if feat_key in data and label_key in data:
            return data[feat_key], data[label_key]
    
    # If no standard pairs found, look for any likely feature/label pair
    potential_features = []
    potential_labels = []
    
    for key, value in data.items():
        if key.startswith('__'):  # Skip metadata
            continue
        if not isinstance(value, np.ndarray):
            continue
            
        # Guess based on shape and name
        if len(value.shape) <= 2:  # Features or labels should be 1D or 2D
            if 'label' in key.lower() or 'target' in key.lower() or 'class' in key.lower():
                potential_labels.append((key, value))
            else:
                potential_features.append((key, value))
    
    if potential_features and potential_labels:
        feat_key, features = potential_features[0]
        label_key, labels = potential_labels[0]
        print(f"\nUsing '{feat_key}' as features and '{label_key}' as labels")
        return features, labels
    
    raise ValueError("Could not find feature and label data in the .mat file")

def load_and_preprocess_data(file_path):
    """Load and preprocess data from .mat file"""
    print(f"Loading data from {file_path}")
    data = loadmat(file_path)
    
    # Print available keys in the .mat file
    print("\nAvailable keys in the data:")
    for key in data.keys():
        if not key.startswith('__'):  # Skip metadata
            value = data[key]
            shape_info = value.shape if hasattr(value, 'shape') else 'No shape'
            dtype_info = value.dtype if hasattr(value, 'dtype') else 'No dtype'
            print(f"- {key}: Shape={shape_info}, Type={dtype_info}")
    
    # Extract features and labels
    X, y = get_data_from_mat(data)
    
    # Ensure correct shapes
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(y.shape) > 1:
        y = y.reshape(-1)
    
    # Convert to appropriate types
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    
    # Normalize labels to start from 0
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    print("\nLabel mapping:")
    for original, new in label_map.items():
        print(f"Original label {original} -> New label {new}")
    
    # Print detailed dataset information
    print(f"\nDataset Information:")
    print("-" * 50)
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    unique_labels = np.unique(y)
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Classes: {unique_labels}")
    
    print("\nClass distribution:")
    for label in unique_labels:
        count = np.sum(y == label)
        percentage = (count / len(y)) * 100
        print(f"Class {label}: {count} samples ({percentage:.1f}%)")
    
    print("\nFeature statistics:")
    print(f"Mean value: {X.mean():.3f}")
    print(f"Std deviation: {X.std():.3f}")
    print(f"Min value: {X.min():.3f}")
    print(f"Max value: {X.max():.3f}")
    print(f"Missing values: {np.isnan(X).sum()}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def main():
    # List all .mat files in data directory with dataset info
    data_dir = './data'
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    print("Available .mat files:")
    print("-" * 50)
    print(f"{'#':<3} {'Dataset':<30} {'Samples':<10} {'Features':<10} {'Classes':<10}")
    print("-" * 50)
    
    dataset_info = []
    for i, file in enumerate(mat_files):
        try:
            data = loadmat(os.path.join(data_dir, file))
            X, y = get_data_from_mat(data)
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            if len(y.shape) > 1:
                y = y.reshape(-1)
            
            info = {
                'name': file,
                'samples': X.shape[0],
                'features': X.shape[1],
                'classes': len(np.unique(y))
            }
            dataset_info.append(info)
            
            print(f"{i+1:<3} {info['name']:<30} {info['samples']:<10} "
                  f"{info['features']:<10} {info['classes']:<10}")
        except Exception as e:
            print(f"{i+1:<3} {file:<30} Error loading dataset: {str(e)}")
    
    # Let user select a file
    while True:
        try:
            selection = int(input("\nSelect a file number to process (or 0 to exit): ")) - 1
            if selection == -1:
                return
            if 0 <= selection < len(mat_files):
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    file_path = os.path.join(data_dir, mat_files[selection])
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    
    # Split data into train, validation, and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    
    print("\nFinal dataset splits:")
    print(f"Training set: {X_train.shape} samples")
    print(f"Validation set: {X_val.shape} samples")
    print(f"Test set: {X_test.shape} samples")
    
    # Confirm before starting search
    input("\nPress Enter to start architecture search, or Ctrl+C to abort...")
    
    # Run architecture search
    search, analysis = run_architecture_search(
        train_data=X_train,
        train_labels=y_train,
        val_data=X_val,
        val_labels=y_val,
        test_data=X_test,
        test_labels=y_test
    )
    
    # Save results with dataset name
    dataset_name = os.path.splitext(mat_files[selection])[0]
    search.save_results(f'afs_results_{dataset_name}.json')

if __name__ == '__main__':
    main()