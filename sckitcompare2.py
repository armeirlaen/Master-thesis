import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (load_breast_cancer, load_digits, 
                            load_diabetes, fetch_olivetti_faces)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif,
                                     SelectFromModel, RFE)
from sklearn.pipeline import Pipeline
from scipy.io import loadmat
import os
import time

class RandomSelector:
    def __init__(self, k):
        self.k = k
        self.selected_features_ = None
        
    def fit(self, X, y):
        n_features = X.shape[1]
        self.selected_features_ = np.random.choice(n_features, self.k, replace=False)
        return self
        
    def transform(self, X):
        return X[:, self.selected_features_]
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices=False):
        support = np.zeros(X.shape[1], dtype=bool)
        support[self.selected_features_] = True
        return support

def evaluate_feature_selection(X, y, fs_method, n_features):
    selector = fs_method(n_features)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', selector),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy')
    return scores.mean(), scores.std()

def plot_feature_index_comparison(X, y, feature_selectors, n_features, dataset_name):
    """
    Plot feature indices selected by Random Forest Importance vs other methods.
    """
    # Get selected features for each method
    selected_features = {}
    rf_selector = feature_selectors['Random Forest Importance'](n_features)
    
    # Fit RF selector first
    if hasattr(rf_selector, 'fit'):
        rf_selector.fit(X, y)
    rf_features = np.where(rf_selector.get_support())[0]
    
    # Get other methods' selected features
    other_methods = {k: v for k, v in feature_selectors.items() 
                    if k != 'Random Forest Importance'}
    
    # Create subplots
    n_methods = len(other_methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, selector_factory) in enumerate(other_methods.items()):
        selector = selector_factory(n_features)
        if hasattr(selector, 'fit'):
            selector.fit(X, y)
        method_features = np.where(selector.get_support())[0]
        
        # Create scatter plot
        axes[idx].scatter(rf_features, method_features, alpha=0.6)
        
        # Add diagonal line for reference
        max_idx = max(X.shape[1], max(rf_features), max(method_features))
        axes[idx].plot([0, max_idx], [0, max_idx], 'r--', alpha=0.5, 
                      label='Perfect Agreement')
        
        # Customize plot
        axes[idx].set_xlabel('Feature Indices (Random Forest Importance)')
        axes[idx].set_ylabel(f'Feature Indices ({method_name})')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
        
        # Add correlation coefficient
        correlation = np.corrcoef(rf_features, method_features)[0,1]
        axes[idx].set_title(f'{method_name} vs Random Forest Importance\n'
                          f'Correlation: {correlation:.3f}')
    
    plt.suptitle(f'Feature Selection Comparison - {dataset_name}\n'
                f'(n_features={n_features})', y=1.02)
    plt.tight_layout()
    plt.show()

# Define feature selection methods
def create_f_score(k):
    return SelectKBest(score_func=f_classif, k=k)

def create_mutual_info(k):
    return SelectKBest(score_func=mutual_info_classif, k=k)

def create_rf_importance(k):
    return SelectFromModel(
        RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        max_features=k,
    )

def create_rfe(k):
    step = max(k // 10, 1)
    return RFE(
        estimator=LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        ),
        n_features_to_select=k,
        step=step
    )

def create_random(k):
    return RandomSelector(k)

feature_selectors = {
    'F-Score': create_f_score,
    'Mutual Information': create_mutual_info,
    'Random Forest Importance': create_rf_importance,
    #'Recursive Feature Elimination': create_rfe,
    'Random Selection': create_random
}

def calculate_jaccard_similarity(selector1, selector2):
    """Calculate Jaccard similarity between two sets of selected features."""
    features1 = set(np.where(selector1.get_support())[0])
    features2 = set(np.where(selector2.get_support())[0])
    
    intersection = len(features1.intersection(features2))
    union = len(features1.union(features2))
    
    return intersection / union if union > 0 else 0

def load_mat_dataset(filepath):
    """
    Load a .mat file and return data and target arrays.
    Assumes the .mat file contains 'X' for features and 'Y' for labels.
    """
    mat_data = loadmat(filepath)
    X = mat_data['X']  # Features
    y = mat_data['Y'].ravel()  # Labels
    return X, y

datasets = {}
data_folder = './data'

for filename in os.listdir(data_folder):
    if filename.endswith('.mat'):
        dataset_name = filename.replace('.mat', '')
        filepath = os.path.join(data_folder, filename)
        try:
            X, y = load_mat_dataset(filepath)
            datasets[dataset_name] = {
                'data': X,
                'target': y,
                'filename': filename
            }
            print(f"Successfully loaded {dataset_name}")
            print(f"Shape: {X.shape}")
            print(f"Number of classes: {len(np.unique(y))}")
            print("-" * 50)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

    # Calculate feature counts using linear spacing
    max_features = min(X.shape[1], 100)
    feature_counts = np.unique(np.linspace(2, max_features, 20).astype(int))
    results = {name: {'mean': [], 'std': [], 'time': []} for name in feature_selectors}
    
    # Add Jaccard similarity tracking
    jaccard_similarities = {name: [] for name in feature_selectors 
                          if name != 'Random Forest Importance'}
    
    # Evaluate methods
    for n_features in feature_counts:
        print(f"\nEvaluating with {n_features} features:")
        
        # Fit Random Forest selector first
        rf_selector = feature_selectors['Random Forest Importance'](n_features)
        rf_selector.fit(X, y)
        
        for name, selector_factory in feature_selectors.items():
            start_time = time.time()
            mean_score, std_score = evaluate_feature_selection(X, y, selector_factory, n_features)
            end_time = time.time()
            
            results[name]['mean'].append(mean_score)
            results[name]['std'].append(std_score)
            results[name]['time'].append(end_time - start_time)
            
            print(f"{name}: {mean_score:.4f} (±{std_score:.4f}) - Time: {end_time - start_time:.2f}s")
            # Calculate Jaccard similarity with RF (if not RF itself)
            if name != 'Random Forest Importance':
                selector = selector_factory(n_features)
                selector.fit(X, y)
                similarity = calculate_jaccard_similarity(rf_selector, selector)
                jaccard_similarities[name].append(similarity)
    
    # Plot original accuracy results
    plt.figure(figsize=(12, 8))
    for name in feature_selectors:
        means = results[name]['mean']
        stds = results[name]['std']
        if name == 'Random Selection':
            plt.plot(feature_counts, means, label=name, marker='o', linestyle='--')
        else:
            plt.plot(feature_counts, means, label=name, marker='o')
        plt.fill_between(feature_counts,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2)
    
    plt.xlabel('Number of Features')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Dataset: {dataset_name}\n'
             f'(n={X.shape[0]} samples, p={X.shape[1]} features)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot Jaccard similarities
    plt.figure(figsize=(12, 8))
    for name, similarities in jaccard_similarities.items():
        plt.plot(feature_counts, similarities, label=name, marker='o')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Jaccard Similarity with Random Forest')
    plt.title(f'Feature Selection Similarity - {dataset_name}\n'
             f'(Comparison with Random Forest Importance)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Rest of the code remains the same...
    
    # Find optimal number of features for each method
    print(f"\nOptimal features for dataset: {dataset_name}")
    optimal_features = {}
    for name in feature_selectors:
        means = results[name]['mean']
        best_n_features = feature_counts[np.argmax(means)]
        best_score = max(means)
        optimal_features[name] = {
            'n_features': best_n_features,
            'score': best_score
        }
        print(f"\n{name}:")
        print(f"Optimal number of features: {best_n_features}")
        print(f"Best balanced accuracy: {best_score:.4f}")

# Print final summary
print("\nOverall Performance Summary:")
print("-" * 50)
for dataset_name in datasets:
    print(f"\n{dataset_name}:")
    dataset_shape = datasets[dataset_name].data.shape
    print(f"Shape: {dataset_shape} (n_samples, n_features)")
    print(f"Feature-to-sample ratio: {dataset_shape[1]/dataset_shape[0]:.3f}")