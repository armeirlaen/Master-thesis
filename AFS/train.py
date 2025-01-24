import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from afs_model import AFS

class AFSTrainer:
    def __init__(self, 
                 input_size,
                 num_classes=10,
                 e_node=32,
                 a_node=2,
                 hidden_size=500,
                 learning_rate=0.8,
                 weight_decay=0.0001,
                 batch_size=100,
                 num_epochs=30,
                 device=None):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize model
        self.model = AFS(input_size, num_classes, e_node, a_node, hidden_size).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data, train_labels, val_data, val_labels):
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data),
            torch.LongTensor(train_labels)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(val_data),
            torch.LongTensor(val_labels)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs, _ = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Validate
            if (epoch + 1) % 1 == 0:
                val_loss, val_acc = self.evaluate(val_loader)
                print(f'Epoch [{epoch+1}/{self.num_epochs}] - '
                      f'Loss: {total_loss/len(train_loader):.6f} - '
                      f'Val Loss: {val_loss:.6f} - '
                      f'Val Acc: {val_acc:.4f}')
            
            self.scheduler.step()
        
        # Get final feature weights
        feature_weights = self.model.get_feature_weights(train_loader, self.device)
        return feature_weights.cpu().numpy()

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs, _ = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        return total_loss/len(loader), correct/total

def test_selected_features(train_data, train_labels, test_data, test_labels, feature_indices,
                         hidden_size=500, num_classes=10, device=None):
    """Test accuracy using only selected features"""
    device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Select features
    train_x = train_data[:, feature_indices]
    test_x = test_data[:, feature_indices]
    
    # Create model for testing
    model = nn.Sequential(
        nn.Linear(len(feature_indices), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    ).to(device)
    
    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_x),
        torch.LongTensor(train_labels)
    )
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    # Train
    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
    # Test
    model.eval()
    test_x = torch.FloatTensor(test_x).to(device)
    test_y = torch.LongTensor(test_labels).to(device)
    
    with torch.no_grad():
        outputs = model(test_x)
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(test_y).float().mean().item()
    
    return accuracy

def run_feature_selection(train_data, train_labels, test_data, test_labels,
                         val_data, val_labels, input_size, num_classes=10):
    """Run complete feature selection process"""
    
    # Initialize trainer
    trainer = AFSTrainer(
        input_size=input_size,
        num_classes=num_classes
    )
    
    # Train and get feature weights
    feature_weights = trainer.train(train_data, train_labels, val_data, val_labels)
    
    # Test different numbers of selected features
    results = []
    for k in range(5, 300, 10):
        # Get top k features
        top_k_indices = np.argsort(feature_weights)[-k:]
        
        # Test accuracy with selected features
        accuracy = test_selected_features(
            train_data, train_labels,
            test_data, test_labels,
            top_k_indices
        )
        
        results.append((k, accuracy))
        print(f'Using top {k} features | Accuracy: {accuracy:.4f}')
    
    return results, feature_weights