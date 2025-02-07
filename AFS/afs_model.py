import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_size, e_node=32, a_node=2):
        super().__init__()
        self.input_size = input_size
        
        # Extract network (E)
        self.extract = nn.Sequential(
            nn.Linear(input_size, e_node),
            nn.Tanh()
        )
        
        # Attention networks for each feature
        self.attention_nets =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(e_node, a_node),
                nn.Linear(a_node, 2)
            )
            for _ in range(input_size)
        ])

    def forward(self, x):
        # Extract features
        e = self.extract(x)  # [batch_size, e_node]
        
        # Calculate attention weights for each feature
        attention_weights = []
        for i in range(self.input_size):
            # Get attention scores for this feature
            attention_scores = self.attention_nets[i](e)  # [batch_size, 2]
            attention_probs = F.softmax(attention_scores, dim=1)
            # Only keep the "select" probability
            attention_weight = attention_probs[:, 1:2]  # [batch_size, 1]
            attention_weights.append(attention_weight)
            
        # Stack all attention weights
        attention = torch.cat(attention_weights, dim=1)  # [batch_size, input_size]
        return attention

class LearningModule(nn.Module):
    def __init__(self, input_size, hidden_size=500, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AFS(nn.Module):
    def __init__(self, input_size, num_classes=10, e_node=32, a_node=2, hidden_size=500):
        super().__init__()
        self.attention_module = AttentionModule(input_size, e_node, a_node)
        self.learning_module = LearningModule(input_size, hidden_size, num_classes)

    def forward(self, x):
        # Generate attention weights
        attention = self.attention_module(x)
        
        # Apply attention weights to input
        weighted_input = x * attention
        
        # Pass through learning module
        output = self.learning_module(weighted_input)
        return output, attention

    def get_feature_weights(self, loader, device):
        """Calculate average attention weights across the dataset"""
        self.eval()
        attention_sum = torch.zeros(self.attention_module.input_size).to(device)
        total_samples = 0
        
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                batch_size = x.size(0)
                _, attention = self(x)
                attention_sum += attention.sum(dim=0)
                total_samples += batch_size
                
        feature_weights = attention_sum / total_samples
        return feature_weights