import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class VIFFeatureSelector:
    """
    Feature selection using Variance Inflation Factor (VIF) to detect multicollinearity
    """
    
    def __init__(self, threshold=5.0):
        """
        Initialize VIF Feature Selector
        
        Parameters:
        threshold (float): VIF threshold above which features are considered multicollinear
        """
        self.threshold = threshold
        self.selected_features = None
        self.vif_results = None
        self.scaler = StandardScaler()
        
    def calculate_vif(self, X):
        """
        Calculate VIF for all features
        
        Parameters:
        X (DataFrame): Feature matrix
        
        Returns:
        DataFrame: VIF results with features, VIF values, and tolerance
        """
        # Standardize features for VIF calculation
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        vif_data = []
        for i, feature in enumerate(X_scaled_df.columns):
            try:
                vif_value = variance_inflation_factor(X_scaled_df.values, i)
                tolerance = 1 / vif_value if vif_value > 0 else np.inf
                vif_data.append({
                    'Feature': feature,
                    'VIF': vif_value,
                    'Tolerance': tolerance
                })
            except:
                # Handle cases where VIF cannot be calculated
                vif_data.append({
                    'Feature': feature,
                    'VIF': np.inf,
                    'Tolerance': 0.0
                })
        
        return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    def select_features(self, X, verbose=True):
        """
        Select features by iteratively removing highest VIF features
        
        Parameters:
        X (DataFrame): Feature matrix
        verbose (bool): Print progress information
        
        Returns:
        DataFrame: Selected features with VIF < threshold
        """
        X_current = X.copy()
        iteration = 0
        
        if verbose:
            print("=== VIF Feature Selection Process ===\n")
        
        while True:
            iteration += 1
            
            # Calculate VIF for current features
            vif_df = self.calculate_vif(X_current)
            
            if verbose:
                print(f"Iteration {iteration}:")
                print(f"Number of features: {len(X_current.columns)}")
                print(f"Max VIF: {vif_df['VIF'].max():.6f}")
                print(f"Features with VIF > {self.threshold}: {sum(vif_df['VIF'] > self.threshold)}")
                print()
            
            # Check if all VIF values are below threshold
            if vif_df['VIF'].max() <= self.threshold:
                if verbose:
                    print(f"✅ All features have VIF ≤ {self.threshold}")
                break
            
            # Find feature with highest VIF
            highest_vif_feature = vif_df.iloc[0]['Feature']
            highest_vif_value = vif_df.iloc[0]['VIF']
            
            if verbose:
                print(f"Removing feature: {highest_vif_feature} (VIF: {highest_vif_value:.6f})")
            
            # Remove the feature with highest VIF
            X_current = X_current.drop(columns=[highest_vif_feature])
            
            # Safety check to prevent infinite loop
            if len(X_current.columns) == 0:
                print("⚠️ Warning: All features removed!")
                break
        
        self.selected_features = X_current.columns.tolist()
        self.vif_results = vif_df
        
        return X_current
    
    def display_vif_results(self, title="VIF Analysis Results"):
        """
        Display VIF results in a formatted table
        """
        if self.vif_results is not None:
            print(f"\n=== {title} ===")
            print(f"{'Feature':<15} {'VIF':<12} {'Tolerance':<12}")
            print("-" * 40)
            
            for _, row in self.vif_results.iterrows():
                vif_val = f"{row['VIF']:.6f}" if row['VIF'] != np.inf else "∞"
                tol_val = f"{row['Tolerance']:.6f}" if row['Tolerance'] != 0 else "0.000000"
                print(f"{row['Feature']:<15} {vif_val:<12} {tol_val:<12}")
        else:
            print("No VIF results available. Run feature selection first.")
    
    def plot_vif_comparison(self, X_original, X_selected):
        """
        Plot VIF comparison before and after feature selection
        """
        # Calculate VIF for original data
        vif_original = self.calculate_vif(X_original)
        
        # Calculate VIF for selected features
        vif_selected = self.calculate_vif(X_selected)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original VIF plot
        ax1.barh(vif_original['Feature'], vif_original['VIF'])
        ax1.axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold}')
        ax1.set_xlabel('VIF')
        ax1.set_title(f'Original Features VIF\n({len(vif_original)} features)')
        ax1.legend()
        ax1.set_xscale('log')
        
        # Selected features VIF plot
        ax2.barh(vif_selected['Feature'], vif_selected['VIF'])
        ax2.axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold}')
        ax2.set_xlabel('VIF')
        ax2.set_title(f'Selected Features VIF\n({len(vif_selected)} features)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function demonstrating VIF feature selection on the provided dataset
    """
    
   
    # Create DataFrame
    df = pd.read_csv("../Train_knight.csv")
    
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Separate features and target
    X = df.drop('knight', axis=1)  # Remove target variable
    y = df['knight']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable: {y.name}")
    
    # Initialize VIF selector
    vif_selector = VIFFeatureSelector(threshold=5.0)
    
    # Calculate initial VIF
    print("\n" + "="*50)
    print("INITIAL VIF ANALYSIS")
    print("="*50)
    
    initial_vif = vif_selector.calculate_vif(X)
    print(f"{'Feature':<15} {'VIF':<15} {'Tolerance':<15}")
    print("-" * 45)
    
    for _, row in initial_vif.iterrows():
        vif_val = f"{row['VIF']:.6f}" if row['VIF'] != np.inf else "∞"
        tol_val = f"{row['Tolerance']:.6f}" if row['Tolerance'] != 0 else "0.000000"
        print(f"{row['Feature']:<15} {vif_val:<15} {tol_val:<15}")
    
    # Perform feature selection
    print(f"\n" + "="*50)
    print("FEATURE SELECTION PROCESS")
    print("="*50)
    
    X_selected = vif_selector.select_features(X, verbose=True)
    
    # Display final results
    print("\n" + "="*50)
    print("FINAL VIF RESULTS (AFTER SELECTION)")
    print("="*50)
    
    vif_selector.display_vif_results("Selected Features")
    
    print(f"\n" + "="*50)
    print("FEATURE SELECTION SUMMARY")
    print("="*50)
    
    print(f"Original features: {len(X.columns)}")
    print(f"Selected features: {len(X_selected.columns)}")
    print(f"Features removed: {len(X.columns) - len(X_selected.columns)}")
    print(f"Reduction percentage: {((len(X.columns) - len(X_selected.columns)) / len(X.columns)) * 100:.1f}%")
    
    print(f"\nSelected features:")
    for i, feature in enumerate(X_selected.columns, 1):
        print(f"{i:2d}. {feature}")
    
    print(f"\nRemoved features:")
    removed_features = set(X.columns) - set(X_selected.columns)
    for i, feature in enumerate(removed_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Create correlation heatmap for comparison
    print(f"\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlations
    corr_original = X.corr()
    corr_selected = X_selected.corr()
    
    # Find highly correlated pairs in original data
    print("Highly correlated feature pairs in original data (|correlation| > 0.8):")
    high_corr_pairs = []
    for i in range(len(corr_original.columns)):
        for j in range(i+1, len(corr_original.columns)):
            corr_val = corr_original.iloc[i, j]
            if abs(corr_val) > 0.8:
                feature1 = corr_original.columns[i]
                feature2 = corr_original.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_val))
                print(f"  {feature1} ↔ {feature2}: {corr_val:.3f}")
    
    if not high_corr_pairs:
        print("  No highly correlated pairs found (threshold: |correlation| > 0.8)")
    
    # Check remaining correlations in selected features
    print(f"\nHighly correlated pairs remaining in selected features:")
    remaining_high_corr = []
    for i in range(len(corr_selected.columns)):
        for j in range(i+1, len(corr_selected.columns)):
            corr_val = corr_selected.iloc[i, j]
            if abs(corr_val) > 0.8:
                feature1 = corr_selected.columns[i]
                feature2 = corr_selected.columns[j]
                remaining_high_corr.append((feature1, feature2, corr_val))
                print(f"  {feature1} ↔ {feature2}: {corr_val:.3f}")
    
    if not remaining_high_corr:
        print("  ✅ No highly correlated pairs remaining!")
    
    # Create visualization function
    def create_correlation_heatmaps():
        """Create side-by-side correlation heatmaps"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Original features correlation heatmap
        mask1 = np.triu(np.ones_like(corr_original, dtype=bool))
        sns.heatmap(corr_original, mask=mask1, annot=False, cmap='coolwarm', 
                   center=0, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title(f'Original Features Correlation\n({len(X.columns)} features)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # Selected features correlation heatmap
        mask2 = np.triu(np.ones_like(corr_selected, dtype=bool))
        sns.heatmap(corr_selected, mask=mask2, annot=True, cmap='coolwarm', 
                   center=0, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title(f'Selected Features Correlation\n({len(X_selected.columns)} features)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    # Uncomment the line below to show correlation heatmaps
    # create_correlation_heatmaps()
    
    return X_selected, vif_selector

# Example usage and analysis
if __name__ == "__main__":
    # Run the feature selection
    X_selected, selector = main()
    
    # Additional analysis
    print(f"\n" + "="*50)
    print("ADDITIONAL INSIGHTS")
    print("="*50)
    
    print("VIF Interpretation:")
    print("• VIF = 1: No multicollinearity")
    print("• 1 < VIF < 5: Moderate multicollinearity")
    print("• 5 ≤ VIF < 10: High multicollinearity") 
    print("• VIF ≥ 10: Very high multicollinearity")
    print("\nTolerance = 1/VIF")
    print("• Tolerance close to 0: High multicollinearity")
    print("• Tolerance close to 1: Low multicollinearity")