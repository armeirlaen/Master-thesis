import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLinearLayer(nn.Module):
    """Configurable non-linear activation layer"""
    def __init__(self, activation_type='tanh'):
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

class AttentionModule(nn.Module):
    def __init__(self, input_size, e_node=32, a_node=2, 
                 extract_activation='tanh', attention_activation='tanh'):
        super().__init__()
        self.input_size = input_size
        
        # Extract network (E) with configurable activation
        self.extract = nn.Sequential(
            nn.Linear(input_size, e_node),
            NonLinearLayer(extract_activation)
        )
        
        # Attention networks with configurable activation
        self.attention_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(e_node, a_node),
                NonLinearLayer(attention_activation)
            )
            for _ in range(input_size)
        ])
    
    def forward(self, x):
        # Extract features
        e = self.extract(x)
        
        # Calculate attention weights for each feature
        attention_weights = []
        for i in range(self.input_size):
            attention_scores = self.attention_nets[i](e)
            attention_probs = F.softmax(attention_scores, dim=1)
            attention_weight = attention_probs[:, 1:2]
            attention_weights.append(attention_weight)
        
        attention = torch.cat(attention_weights, dim=1)
        return attention

class LearningModule(nn.Module):
    def __init__(self, input_size, hidden_size=500, num_classes=10,
                 hidden_activation='relu', dropout_rate=0.0):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            NonLinearLayer(hidden_activation),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class AFS(nn.Module):
    def __init__(self, input_size, num_classes=10, e_node=32, a_node=2, 
                 hidden_size=500, extract_activation='tanh',
                 attention_activation='tanh', hidden_activation='relu',
                 dropout_rate=0.0):
        super().__init__()
        
        self.attention_module = AttentionModule(
            input_size=input_size,
            e_node=e_node,
            a_node=a_node,
            extract_activation=extract_activation,
            attention_activation=attention_activation
        )
        
        self.learning_module = LearningModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            hidden_activation=hidden_activation,
            dropout_rate=dropout_rate
        )
    
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