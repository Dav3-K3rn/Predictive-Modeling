# =============================================================================
# IEEE Research Paper Demo: Predictive Modeling in Python
# FIXED VERSION with Improved Plotting
# =============================================================================

# STEP 1: Environment Setup with Version Checking
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("IEEE PREDICTIVE MODELING DEMO")
print("=" * 70)

# Check Python version
print(f"Python version: {sys.version}")
print()

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')  # Or 'Qt5Agg' or 'Agg' for different backends
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.datasets import load_breast_cancer, load_diabetes, make_blobs
    
    # Import interpretation library (optional)
    try:
        import shap
        SHAP_AVAILABLE = True
        print("SHAP installed - Model interpretation enabled")
    except ImportError:
        SHAP_AVAILABLE = False
        print("SHAP not installed - Install with: pip install shap")
    
    print("All required packages imported successfully")
    
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Please install with: pip install pandas numpy matplotlib seaborn scikit-learn")
    sys.exit(1)

# Set better plotting style
plt.style.use('default')
sns.set_palette("husl")

# STEP 2: Data Loading - Using reliable datasets
def load_standard_datasets():
    """
    Load standard sklearn datasets for reliable demonstration
    """
    print("\nLoading standard datasets for analysis...")
    
    # Load multiple standard datasets
    cancer_data = load_breast_cancer()
    diabetes_data = load_diabetes()
    
    # Create DataFrames
    df_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    df_cancer['diagnosis'] = cancer_data.target
    df_cancer['diagnosis'] = df_cancer['diagnosis'].map({0: 'malignant', 1: 'benign'})
    
    df_diabetes = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
    df_diabetes['target'] = diabetes_data.target
    
    # Create clustering dataset
    X_cluster, y_cluster = make_blobs(n_samples=300, centers=3, n_features=2, 
                                    random_state=42, cluster_std=1.0)
    df_cluster = pd.DataFrame(X_cluster, columns=['Feature_1', 'Feature_2'])
    df_cluster['true_cluster'] = y_cluster
    
    return {
        'classification': df_cancer,
        'regression': df_diabetes, 
        'clustering': df_cluster
    }

# Load the data
datasets = load_standard_datasets()

# STEP 3: Simplified EDA Function
def perform_basic_eda(df, dataset_name):
    """
    Perform basic exploratory data analysis
    """
    print(f"\n" + "="*50)
    print(f"EXPLORATORY DATA ANALYSIS: {dataset_name.upper()}")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Create a simple visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: For classification - target distribution
    if 'diagnosis' in df.columns:
        plt.subplot(1, 3, 1)
        df['diagnosis'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Distribution')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
    
    # Plot 2: For regression - target distribution
    elif 'target' in df.columns:
        plt.subplot(1, 3, 1)
        plt.hist(df['target'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.title('Target Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')
    
    # Plot 3: Correlation heatmap for numeric features
    plt.subplot(1, 3, 2)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        # Use only first 10 features for readability
        corr_matrix = numeric_df.iloc[:, :10].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap (First 10 features)')
    
    # Plot 4: Feature distribution
    plt.subplot(1, 3, 3)
    if len(numeric_df.columns) > 0:
        # Plot distribution of first numeric feature
        plt.hist(numeric_df.iloc[:, 0], bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'Distribution of {numeric_df.columns[0]}')
        plt.xlabel(numeric_df.columns[0])
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return df

# STEP 4: Classification Modeling (Simplified)
def run_classification_demo():
    """
    Run classification analysis
    """
    print("\n" + "="*50)
    print("CLASSIFICATION MODELING - Breast Cancer Dataset")
    print("="*50)
    
    df = datasets['classification']
    df_clean = perform_basic_eda(df, "Breast Cancer Classification")
    
    # Prepare data
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Create comparison visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Model Comparison
    plt.subplot(1, 3, 1)
    model_names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), f1_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Comparison (F1-Scores)')
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(i, bar.get_height() + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 2: Confusion Matrix for best model
    plt.subplot(1, 3, 2)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_scaled)
    
    cm = metrics.confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 3: ROC Curves
    plt.subplot(1, 3, 3)
    for name, result in results.items():
        y_pred_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n  Best Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
    
    return results

# STEP 5: Regression Modeling (Simplified)
def run_regression_demo():
    """
    Run regression analysis
    """
    print("\n" + "="*50)
    print("REGRESSION MODELING - Diabetes Dataset")
    print("="*50)
    
    df = datasets['regression']
    df_clean = perform_basic_eda(df, "Diabetes Regression")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    # Create comparison visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Model Comparison (R² scores)
    plt.subplot(1, 3, 1)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Comparison (R² Scores)')
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        plt.text(i, bar.get_height() + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 2: Predicted vs Actual for best model
    plt.subplot(1, 3, 2)
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_scaled)
    
    plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual - {best_model_name}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals for best model
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred_best
    plt.scatter(y_pred_best, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {best_model_name}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n  Best Model: {best_model_name} (R²: {results[best_model_name]['r2']:.4f})")
    
    return results

# STEP 6: Clustering Demo (Simplified)
def run_clustering_demo():
    """
    Run clustering analysis
    """
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS - Customer Segmentation")
    print("="*50)
    
    df = datasets['clustering']
    df_clean = perform_basic_eda(df, "Customer Clustering")
    
    # Prepare data
    X = df[['Feature_1', 'Feature_2']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette_kmeans = metrics.silhouette_score(X_scaled, kmeans_labels)
    
    print(f"K-Means Silhouette Score: {silhouette_kmeans:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: True clusters
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(df['Feature_1'], df['Feature_2'], c=df['true_cluster'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('True Clusters (Ground Truth)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: K-Means results
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(df['Feature_1'], df['Feature_2'], c=kmeans_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'K-Means Clustering\n(Silhouette: {silhouette_kmeans:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, alpha=0.8, label='Centroids')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Elbow method for optimal k
    plt.subplot(1, 3, 3)
    inertias = []
    K = range(1, 8)
    for k in K:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)
    
    plt.plot(K, inertias, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {'kmeans': kmeans, 'labels': kmeans_labels, 'silhouette': silhouette_kmeans}

# STEP 7: Main Execution
def main():
    """
    Main function to run the complete predictive modeling demo
    """
    print("\n" + "="*70)
    print("STARTING PREDICTIVE MODELING DEMONSTRATION")
    print("="*70)
    
    try:
        # Run all three modeling tasks
        classification_results = run_classification_demo()
        regression_results = run_regression_demo()
        clustering_results = run_clustering_demo()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Classification: Breast Cancer Diagnosis")
    print("Regression: Diabetes Progression") 
    print("Clustering: Customer Segmentation")
    print("Comprehensive visualizations generated")
    print("Performance metrics calculated")
    print("\nThis demo provides a complete framework for IEEE research paper.")
    print("All code is reproducible and ready for academic publication.")

# Run the main function
if __name__ == "__main__":
    main()
