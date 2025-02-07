import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from train import run_feature_selection
from sklearn.model_selection import train_test_split

def load_mnist_mat():
    """Load MNIST dataset from .mat file"""
    try:
        print("Loading MNIST from /data/mnist.mat...")
        data = loadmat('./data/mnist.mat')
        
        # Extract and transpose data to get correct shape
        X = data['data'].T.astype(np.float32)  # Transpose to get [n_samples, n_features]
        y = data['label'].reshape(-1).astype(np.int64)  # Reshape to 1D array
        
        # Split into train and test sets (80-20 split)
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create validation set from training data (10% of original data)
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=0.125, random_state=42, stratify=train_labels
        )
        
        # Verify data shapes and content
        print("\nDataset Information:")
        print("-" * 50)
        print(f"Train data shape: {train_data.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Validation labels shape: {val_labels.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        
        # Check value ranges
        print("\nValue Ranges:")
        print("-" * 50)
        print(f"Train data range: [{train_data.min():.2f}, {train_data.max():.2f}]")
        print(f"Test data range: [{test_data.min():.2f}, {test_data.max():.2f}]")
        print(f"Unique labels in train: {np.unique(train_labels)}")
        print(f"Label distribution in train:")
        for label in np.unique(train_labels):
            count = np.sum(train_labels == label)
            print(f"  Label {label}: {count} samples")
        
        # Normalize data if needed
        if train_data.max() > 1:
            print("\nNormalizing data to [0,1] range...")
            train_data = train_data / 255.0
            val_data = val_data / 255.0
            test_data = test_data / 255.0
        
        print("\nFinal Dataset Splits:")
        print("-" * 50)
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Basic sanity checks
        assert not np.isnan(train_data).any(), "NaN values found in training data"
        assert not np.isnan(test_data).any(), "NaN values found in test data"
        assert train_data.max() <= 1.0 and train_data.min() >= 0.0, "Train data not in [0,1] range"
        assert test_data.max() <= 1.0 and test_data.min() >= 0.0, "Test data not in [0,1] range"
        
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def plot_results(results):
    """Plot accuracy vs number of features"""
    features, accuracies = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(features, accuracies, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Feature Selection Results')
    plt.grid(True)
    plt.savefig('afs_results.png')
    plt.close()
    
    # Also save numerical results
    np.savetxt('feature_selection_results.csv', 
               np.column_stack((features, accuracies)), 
               delimiter=',', 
               header='features,accuracy',
               comments='')

def visualize_feature_weights(feature_weights):
    """Visualize feature weights with color-coded importance levels"""
    # Get indices sorted by importance
    sorted_indices = np.argsort(feature_weights)[::-1]  # Descending order
    
    # Create a mask array initialized to white (least important)
    colored_weights = np.ones((784, 3))  # RGB array filled with white
    
    # Color code based on importance:
    # Red for top 65
    colored_weights[sorted_indices[:65]] = [1, 0, 0]  # Red
    # Yellow for 65-150
    colored_weights[sorted_indices[65:150]] = [1, 1, 0]  # Yellow
    # Green for 150-300
    colored_weights[sorted_indices[150:300]] = [0, 1, 0]  # Green
    # Rest remains white
    
    # Reshape to 28x28x3 for RGB image
    colored_image = colored_weights.reshape(28, 28, 3)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(colored_image)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Top 65 features'),
        Patch(facecolor='yellow', label='Features 66-150'),
        Patch(facecolor='green', label='Features 151-300'),
        Patch(facecolor='white', edgecolor='black', label='Remaining features')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title('Feature Importance Map')
    plt.axis('off')  # Hide axes
    plt.tight_layout()  # Adjust layout to make room for legend
    plt.savefig('feature_weights.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save feature weights
    np.save('feature_weights.npy', feature_weights)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and verify data
    print("Starting AFS feature selection process...")
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_mnist_mat()
    
    # Confirm with user before proceeding
    input("\nPress Enter to continue with feature selection, or Ctrl+C to abort...")
    
    # Run feature selection
    print("\nRunning feature selection...")
    results, feature_weights = run_feature_selection(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        val_data=val_data,
        val_labels=val_labels,
        input_size=784,  # 28*28 for MNIST
        num_classes=10
    )
    
    # Plot and save results
    print("\nGenerating visualizations...")
    plot_results(results)
    visualize_feature_weights(feature_weights)
    
    # Print final results
    print("\nFeature Selection Results:")
    print("-" * 50)
    print("Features  Accuracy")
    print("-" * 50)
    for num_features, accuracy in results:
        print(f"{num_features:^8d}  {accuracy:^8.4f}")

if __name__ == '__main__':
    main()