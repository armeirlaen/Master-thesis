import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from time import time
import os
import scipy.io
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Import fast skfeature methods
from skfeature.function.similarity_based import fisher_score, reliefF
from skfeature.function.statistical_based import chi_square, f_score
from skfeature.utility.construct_W import construct_W

class FastSKFeatureComparison:
    def __init__(self, n_features_to_select=50, n_splits=5, random_state=42, max_time_per_method=300):
        self.n_features_to_select = n_features_to_select
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_time_per_method = max_time_per_method
        
        # Only use the most robust methods
        self.methods = {
            'Random': self._random_selector,
            'Fisher': self._fisher_score_selector,
            'Chi2': self._chi_square_selector,
            'F-score': self._f_score_selector
            #'ReliefF': self._relief_f_selector
        }
        
        self.classifier = SVC(kernel='linear', random_state=random_state)
    
    def _random_selector(self, X, y):
        """Random feature selection baseline"""
        try:
            # Set random seed for reproducibility but different from other random processes
            rng = np.random.RandomState(self.random_state + 100)
            # Generate random permutation of feature indices
            all_features = np.arange(X.shape[1])
            rng.shuffle(all_features)
            return all_features[:self.n_features_to_select]
        except:
            return None
        
    def _remove_constant_features(self, X):
        """Remove constant features from the dataset"""
        std = np.std(X, axis=0)
        non_constant_mask = std != 0
        return X[:, non_constant_mask], non_constant_mask
        
    def _fisher_score_selector(self, X, y):
        try:
            scores = fisher_score.fisher_score(X, y)
            idx = np.argsort(scores)[::-1]
            return idx[:self.n_features_to_select]
        except:
            return None

    def _chi_square_selector(self, X, y):
        try:
            # Ensure non-negative values
            X = X - X.min()
            scores = chi_square.chi_square(X, y)
            idx = np.argsort(scores)[::-1]
            return idx[:self.n_features_to_select]
        except:
            return None

    def _f_score_selector(self, X, y):
        try:
            scores = f_score.f_score(X, y)
            idx = np.argsort(scores)[::-1]
            return idx[:self.n_features_to_select]
        except:
            return None
    
    def _relief_f_selector(self, X, y):
        try:
            scores = reliefF.reliefF(X, y)
            idx = np.argsort(scores)[::-1]
            return idx[:self.n_features_to_select]
        except:
            return None

    def evaluate_method(self, X, y, method_name, method):
        """Evaluate a single feature selection method"""
        start_time = time()
        
        try:
            # Remove constant features
            X, non_constant_mask = self._remove_constant_features(X)
            if X.shape[1] < self.n_features_to_select:
                return f"Error: Only {X.shape[1]} non-constant features"
            
            selected_features = method(X, y)
            if selected_features is None:
                return "Method failed"
            
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                               random_state=self.random_state)
            
            accuracies = []
            f1_scores = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]
                
                self.classifier.fit(X_train_selected, y_train)
                y_pred = self.classifier.predict(X_test_selected)
                
                accuracies.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            
            exec_time = time() - start_time
            
            if exec_time > self.max_time_per_method:
                return "Timeout"
                
            return {
                'acc': f"{np.mean(accuracies):.3f}±{np.std(accuracies):.3f}",
                'f1': f"{np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}",
                'time': f"{exec_time:.1f}s"
            }
            
        except Exception as e:
            return str(e)[:50]  # Truncate long error messages

def create_comparison_table():
    """Create a comparison table for all datasets and methods"""
    results = []
    
    # Get all .mat files
    dataset_dir = './data'
    datasets = [f[:-4] for f in os.listdir(dataset_dir) if f.endswith('.mat')]
    dataset = datasets[:5]
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        try:
            # Load and preprocess data
            mat = scipy.io.loadmat(f'./data/{dataset}.mat')
            X = mat['X'].astype(float)  # Ensure float type
            y = mat['Y'].ravel()
            
            # Basic dataset information
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))
            
            # Standardize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Initialize comparison object
            fs_comparison = FastSKFeatureComparison(n_features_to_select=50)
            
            # Get results for each method
            method_results = {}
            for method_name, method in fs_comparison.methods.items():
                result = fs_comparison.evaluate_method(X, y, method_name, method)
                if isinstance(result, dict):
                    method_results[method_name] = f"Acc: {result['acc']}\nF1: {result['f1']}\nTime: {result['time']}"
                else:
                    method_results[method_name] = result
                print(method_name,result['acc'],result['f1'],result['time'])
            
            # Combine all information
            result_row = {
                'Dataset': dataset,
                '#Samples': n_samples,
                '#Features': n_features,
                '#Classes': n_classes,
                **method_results
            }
            results.append(result_row)            
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            results.append({
                'Dataset': dataset,
                '#Samples': 'Error',
                '#Features': 'Error',
                '#Classes': 'Error',
                **{method: 'Error' for method in fs_comparison.methods.keys()}
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Create comparison table
    results_table = create_comparison_table()
    
    # Save full results to CSV
    results_table.to_csv('feature_selection_comparison_results.csv', index=False)
    
    # Display formatted table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\nFeature Selection Methods Comparison Results:")
    print("===========================================")
    print(results_table)