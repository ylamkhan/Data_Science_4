import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class ConfusionMatrix:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        unique_labels = sorted(list(set(y_true + y_pred)))
        self.labels = labels if labels else unique_labels
        self.n_classes = len(self.labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.matrix = self._calculate_matrix()
        
    def _calculate_matrix(self):
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            true_idx = self.label_to_idx[true_label]
            pred_idx = self.label_to_idx[pred_label]
            matrix[true_idx][pred_idx] += 1
        return matrix
    
    def calculate_metrics(self):
        metrics = {}
        for i, label in enumerate(self.labels):
            tp = self.matrix[i, i]
            fp = np.sum(self.matrix[:, i]) - tp
            fn = np.sum(self.matrix[i, :]) - tp
            tn = np.sum(self.matrix) - tp - fp - fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            total = np.sum(self.matrix[i, :])
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total': total
            }
        accuracy = np.trace(self.matrix) / np.sum(self.matrix)
        metrics['accuracy'] = accuracy
        metrics['total_samples'] = np.sum(self.matrix)
        return metrics
    
    def print_metrics(self):
        """Print metrics in the required format"""
        metrics = self.calculate_metrics()
        print(f"{'':>12} {'precision':>9} {'recall':>6} {'f1-score':>8} {'total':>5}")
        for label in self.labels:
            m = metrics[label]
            print(f"{label:>12} {m['precision']:>9.2f} {m['recall']:>6.2f} {m['f1_score']:>8.2f} {m['total']:>5d}")
        print(f"{'accuracy':>12} {metrics['accuracy']:>24.2f} {metrics['total_samples']:>5d}")
        print()
        print(self.matrix.tolist())
    
    def plot_confusion_matrix(self, title="Confusion Matrix", figsize=(8, 6)):
        """Display confusion matrix as heatmap"""
        plt.figure(figsize=figsize)
        sns.heatmap(self.matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.labels,
                   yticklabels=self.labels,
                   cbar_kws={'label': 'Count'})
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

def load_file(filename):
    """Load labels from file"""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python Confusion_Matrix.py predictions.txt truth.txt")
        sys.exit(1)
    if sys.argv[1] != "predictions.txt" and sys.argv[2] != "truth.txt" :
        print("Usage: python Confusion_Matrix.py predictions.txt truth.txt")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    truth_file = sys.argv[2]
    y_pred = load_file(predictions_file)
    y_true = load_file(truth_file)
    if len(y_pred) != len(y_true):
        print(f"Error: Number of predictions ({len(y_pred)}) doesn't match number of true labels ({len(y_true)})")
        sys.exit(1)
    if len(y_pred) == 0:
        print("Error: No data found in files")
        sys.exit(1)
    cm = ConfusionMatrix(y_true, y_pred, labels=['Jedi', 'Sith'])
    cm.print_metrics()
    cm.plot_confusion_matrix()

if __name__ == "__main__":
    main()