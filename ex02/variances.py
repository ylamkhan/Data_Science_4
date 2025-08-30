import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import StringIO




def main():
    df = pd.read_csv("../Train_knight.csv")
    df['knight'] = df['knight'].map({'Sith': 1, 'Jedi': 0})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA()
    pca.fit(X_scaled)
    explained_variances = pca.explained_variance_ratio_ * 100
    cumulative_variances = np.cumsum(explained_variances)
    components_for_90_variance = np.argmax(cumulative_variances >= 90) + 1
    print("--- Variance Analysis ---")
    print("\nVariances (Percentage):")
    print(explained_variances)
    print("\nCumulative Variances (Percentage):")
    print(cumulative_variances)
    print(f"\nNumber of components needed to reach 90% variance: {components_for_90_variance}")

    # --- 6. Display Graph ---
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(12, 7))

    # Plot the cumulative variance
    plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o', linestyle='--', label='Cumulative Variance')
    plt.title('Cumulative Explained Variance by Principal Components', fontsize=16, weight='bold')
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance (%)', fontsize=12)

    # Add the 90% threshold line for clarity
    plt.axhline(y=90, color='red', linestyle=':', label='90% Threshold')
    plt.text(1, 91, '90% Variance Threshold', color = 'red', fontsize=11)

    # Mark the point where 90% variance is reached
    plt.axvline(x=components_for_90_variance, color='green', linestyle=':', label=f'{components_for_90_variance} Components for 90%')

    plt.xticks(range(1, len(cumulative_variances) + 1))
    plt.ylim(0, 105)
    plt.legend(loc='center right')
    plt.show()