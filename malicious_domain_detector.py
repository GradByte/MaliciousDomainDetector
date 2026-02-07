"""
Malicious Domain Detection using Machine Learning
Dataset: Mendeley (623sshkdrz) - Benign and malicious domains based on DNS logs
Source: https://data.mendeley.com/datasets/623sshkdrz/5

This model classifies domains as benign (0) or malicious (1) using various features
extracted from DNS logs and domain characteristics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_curve,
    roc_auc_score,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path):
    """
    Load the dataset and preprocess features.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        feature_names (list): List of feature names used
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    print(f"Percentage: {df['Class'].value_counts(normalize=True) * 100}\n")
    
    # Display basic info
    print("Dataset columns:", df.columns.tolist())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df


def select_features(df):
    """
    Select and engineer features for the model.
    
    Feature Selection Rationale:
    
    1. ENTROPY-BASED FEATURES:
       - Entropy: Measures randomness in domain names. Malicious domains often use 
         random character sequences (e.g., "xj3k2m9p.com") leading to high entropy.
       - EntropyOfSubDomains: High entropy in subdomains indicates potential DGA 
         (Domain Generation Algorithm) usage by malware.
    
    2. CHARACTER COMPOSITION FEATURES:
       - ConsoantRatio (typo in dataset): Legitimate domains typically have balanced 
         vowel/consonant ratios. Malicious domains may have unusual patterns.
       - NumericRatio: High numeric content suggests auto-generated domains.
       - SpecialCharRatio: Excessive special characters are uncommon in legitimate domains.
       - VowelRatio: Complement to consonant ratio for linguistic analysis.
       - StrangeCharacters: Count of unusual characters often found in phishing/malicious domains.
    
    3. SEQUENCE FEATURES:
       - ConsoantSequence (typo in dataset): Long consonant sequences (e.g., "xkrtb") 
         are rare in real words.
       - NumericSequence: Long numeric sequences suggest automated generation.
       - SpecialCharSequence: Multiple special chars in sequence indicate suspicious patterns.
    
    4. DOMAIN STRUCTURE FEATURES:
       - DomainLength: Extremely long domains are often malicious (phishing, typosquatting).
       - SubdomainNumber: Multiple subdomains can indicate command & control servers.
    
    5. DNS & SECURITY FEATURES:
       - HasSPFInfo, HasDkimInfo, HasDmarcInfo: Email authentication. Legitimate businesses
         typically implement these; malicious domains often don't.
       - MXDnsResponse: Presence of mail exchange records indicates established infrastructure.
       - TXTDnsResponse: TXT records are used for domain verification.
    
    6. INFRASTRUCTURE FEATURES:
       - DomainInAlexaDB: Presence in Alexa database suggests legitimate, popular domain.
       - CommonPorts: Malicious domains may use non-standard ports.
       - HttpResponseCode: Response codes indicate domain accessibility and behavior.
    
    7. REPUTATION & REGISTRATION FEATURES:
       - IpReputation, DomainReputation: Direct indicators from threat intelligence.
       - CreationDate, LastUpdateDate: Recently created domains are more likely malicious.
       - ASN: Certain ASNs are known to host malicious content.
       - CountryCode: Some countries have higher concentrations of malicious infrastructure.
    
    8. TLD (Top-Level Domain):
       - Certain TLDs (.tk, .ml, .ga) are more commonly abused by attackers due to 
         free registration and lax policies.
    """
    
    # Define features to use
    # Note: Dataset has typos in column names (ConsoantRatio/ConsoantSequence)
    numeric_features = [
        'Entropy',
        'EntropyOfSubDomains',
        'ConsoantRatio',  # Typo in dataset (should be ConsonantRatio)
        'NumericRatio',
        'SpecialCharRatio',
        'VowelRatio',
        'ConsoantSequence',  # Typo in dataset (should be ConsonantSequence)
        'VowelSequence',
        'NumericSequence',
        'SpecialCharSequence',
        'DomainLength',
        'SubdomainNumber',
        'StrangeCharacters',
        'CreationDate',
        'LastUpdateDate',
        'ASN',
        'HttpResponseCode'
    ]
    
    boolean_features = [
        'MXDnsResponse',
        'TXTDnsResponse',
        'HasSPFInfo',
        'HasDkimInfo',
        'HasDmarcInfo',
        'DomainInAlexaDB',
        'CommonPorts',
        'IpReputation',
        'DomainReputation'
    ]
    
    categorical_features = [
        'TLD',
        'CountryCode',
        'DNSRecordType'
    ]
    
    # Create a copy for feature engineering
    df_features = df.copy()
    
    # Convert boolean features to integers (True/False to 1/0)
    for col in boolean_features:
        df_features[col] = df_features[col].astype(str).map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
        df_features[col] = df_features[col].fillna(0).astype(int)
    
    # Handle numeric features
    for col in numeric_features:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        df_features[col] = df_features[col].fillna(df_features[col].median())
    
    # Handle categorical features with label encoding
    label_encoders = {}
    for col in categorical_features:
        df_features[col] = df_features[col].fillna('unknown')
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        label_encoders[col] = le
    
    # Combine all features
    all_features = numeric_features + boolean_features + categorical_features
    
    # Prepare X (features) and y (target)
    X = df_features[all_features]
    y = df['Class']
    
    print(f"\nSelected {len(all_features)} features for the model:")
    print(f"- Numeric features: {len(numeric_features)}")
    print(f"- Boolean features: {len(boolean_features)}")
    print(f"- Categorical features: {len(categorical_features)}")
    
    return X, y, all_features, label_encoders


def train_model(X, y):
    """
    Train a Random Forest classifier.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        model: Trained model
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest model
    # Random Forest is chosen because:
    # 1. Handles mixed feature types well
    # 2. Robust to outliers and missing values
    # 3. Provides feature importance rankings
    # 4. Generally performs well on classification tasks
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle any class imbalance
    )
    
    model.fit(X_train, y_train)
    print("Training completed!")
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate the model and display metrics including AUC-ROC and overfitting analysis.
    
    Args:
        model: Trained model
        X_train, X_test, y_train, y_test: Split datasets
        feature_names: List of feature names
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Prediction probabilities for ROC curve
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    
    # Test accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Calculate AUC-ROC scores
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nTraining AUC-ROC: {train_auc:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")
    
    # Overfitting Detection
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    accuracy_gap = train_accuracy - test_accuracy
    auc_gap = train_auc - test_auc
    
    print(f"Accuracy Gap (Train - Test): {accuracy_gap:.4f}")
    print(f"AUC Gap (Train - Test): {auc_gap:.4f}")
    
    if accuracy_gap < 0.05 and auc_gap < 0.05:
        print("\n✓ Model Status: GOOD - No significant overfitting detected")
        print("  The model generalizes well to unseen data.")
    elif accuracy_gap < 0.10 and auc_gap < 0.10:
        print("\n⚠ Model Status: MODERATE - Slight overfitting detected")
        print("  The model shows some overfitting but is still acceptable.")
    else:
        print("\n✗ Model Status: OVERFITTING - Significant overfitting detected")
        print("  Consider: reducing model complexity, adding regularization,")
        print("  or increasing training data.")
    
    # Cross-validation for robustness check
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS (5-Fold)")
    print("="*60)
    
    # Combine train and test for cross-validation
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    
    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
    cv_auc_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='roc_auc')
    
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"CV Accuracy Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"\nCV AUC Scores: {cv_auc_scores}")
    print(f"CV AUC Mean: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std() * 2:.4f})")
    
    # Check if CV scores are consistent
    cv_variance = cv_scores.std()
    if cv_variance < 0.02:
        print("\n✓ Cross-Validation: STABLE - Model shows consistent performance across folds")
    elif cv_variance < 0.05:
        print("\n⚠ Cross-Validation: MODERATE - Some variance in performance across folds")
    else:
        print("\n✗ Cross-Validation: UNSTABLE - High variance suggests overfitting or data issues")
    
    # Calculate ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*60)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Benign', 'Malicious']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nInterpretation:")
    print(f"True Negatives (Benign correctly classified): {cm[0][0]}")
    print(f"False Positives (Benign incorrectly classified as Malicious): {cm[0][1]}")
    print(f"False Negatives (Malicious incorrectly classified as Benign): {cm[1][0]}")
    print(f"True Positives (Malicious correctly classified): {cm[1][1]}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*60)
    print(feature_importance.head(15).to_string(index=False))
    
    # Return data for plotting
    roc_data = {
        'fpr_train': fpr_train,
        'tpr_train': tpr_train,
        'fpr_test': fpr_test,
        'tpr_test': tpr_test,
        'train_auc': train_auc,
        'test_auc': test_auc
    }
    
    return feature_importance, cm, roc_data


def plot_results(feature_importance, cm, roc_data):
    """
    Create visualizations for model results including ROC curves.
    
    Args:
        feature_importance: DataFrame with feature importance
        cm: Confusion matrix
        roc_data: Dictionary with ROC curve data
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Feature Importance
    top_features = feature_importance.head(15)
    axes[0, 0].barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    axes[0, 0].set_xlabel('Importance Score', fontsize=11)
    axes[0, 0].set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                cbar_kws={'label': 'Count'})
    axes[0, 1].set_xlabel('Predicted Label', fontsize=11)
    axes[0, 1].set_ylabel('True Label', fontsize=11)
    axes[0, 1].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    # Plot 3: ROC Curve
    axes[1, 0].plot(roc_data['fpr_train'], roc_data['tpr_train'], 
                    color='blue', lw=2, label=f'Train ROC (AUC = {roc_data["train_auc"]:.4f})')
    axes[1, 0].plot(roc_data['fpr_test'], roc_data['tpr_test'], 
                    color='red', lw=2, label=f'Test ROC (AUC = {roc_data["test_auc"]:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 0].set_ylabel('True Positive Rate', fontsize=11)
    axes[1, 0].set_title('ROC Curve (Receiver Operating Characteristic)', fontsize=13, fontweight='bold')
    axes[1, 0].legend(loc="lower right", fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Overfitting Analysis (Train vs Test Comparison)
    metrics = ['Accuracy', 'AUC-ROC']
    train_scores = [roc_data.get('train_accuracy', 0), roc_data['train_auc']]
    test_scores = [roc_data.get('test_accuracy', 0), roc_data['test_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, train_scores, width, label='Train', color='blue', alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, test_scores, width, label='Test', color='red', alpha=0.7)
    
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Train vs Test Performance (Overfitting Check)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('malicious_domain_detection_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results visualization saved as 'malicious_domain_detection_results.png'")
    plt.show()


def predict_new_domain(model, label_encoders, feature_names):
    """
    Example function to predict if a new domain is malicious.
    
    Args:
        model: Trained model
        label_encoders: Dictionary of label encoders for categorical features
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("EXAMPLE: Predicting a new domain")
    print("="*60)
    
    # Example domain features (you would extract these from a real domain)
    example_features = {
        'Entropy': 4.5,
        'EntropyOfSubDomains': 0,
        'ConsoantRatio': 0.6,  # Typo in dataset
        'NumericRatio': 0.1,
        'SpecialCharRatio': 0.0,
        'VowelRatio': 0.3,
        'ConsoantSequence': 8,  # Typo in dataset
        'VowelSequence': 2,
        'NumericSequence': 2,
        'SpecialCharSequence': 0,
        'DomainLength': 120,
        'SubdomainNumber': 0,
        'StrangeCharacters': 15,
        'CreationDate': 0,
        'LastUpdateDate': 0,
        'ASN': -1,
        'HttpResponseCode': 0,
        'MXDnsResponse': 0,
        'TXTDnsResponse': 0,
        'HasSPFInfo': 0,
        'HasDkimInfo': 0,
        'HasDmarcInfo': 0,
        'DomainInAlexaDB': 0,
        'CommonPorts': 0,
        'IpReputation': 0,
        'DomainReputation': 0,
        'TLD': 0,  # Would be encoded
        'CountryCode': 0,  # Would be encoded
        'DNSRecordType': 0  # Would be encoded
    }
    
    # Create DataFrame
    X_new = pd.DataFrame([example_features])
    
    # Make prediction
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]
    
    print(f"\nPrediction: {'MALICIOUS' if prediction == 1 else 'BENIGN'}")
    print(f"Confidence: Benign={probability[0]:.2%}, Malicious={probability[1]:.2%}")


def main():
    """
    Main function to run the complete pipeline.
    """
    print("="*60)
    print("MALICIOUS DOMAIN DETECTION MODEL")
    print("="*60)
    
    # File path
    file_path = 'datasets/BenignAndMaliciousDataset.csv'
    
    # Load and preprocess data
    print("\n[1/5] Loading dataset...")
    df = load_and_preprocess_data(file_path)
    
    # Feature selection
    print("\n[2/5] Selecting and engineering features...")
    X, y, feature_names, label_encoders = select_features(df)
    
    # Train model
    print("\n[3/5] Training model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Evaluate model
    print("\n[4/5] Evaluating model...")
    feature_importance, cm, roc_data = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
    
    # Add accuracy to roc_data for plotting
    roc_data['train_accuracy'] = accuracy_score(y_train, model.predict(X_train))
    roc_data['test_accuracy'] = accuracy_score(y_test, model.predict(X_test))
    
    # Plot results
    print("\n[5/5] Creating visualizations...")
    plot_results(feature_importance, cm, roc_data)
    
    # Example prediction
    predict_new_domain(model, label_encoders, feature_names)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("1. The model uses 33 features covering domain structure, DNS records,")
    print("   reputation, and statistical characteristics.")
    print("2. Random Forest classifier provides robust performance and interpretability.")
    print("3. Feature importance helps understand which attributes are most indicative")
    print("   of malicious behavior.")
    print("4. AUC-ROC score measures the model's ability to distinguish between classes.")
    print("   (1.0 = perfect, 0.5 = random guessing)")
    print("5. Cross-validation ensures the model generalizes well to unseen data.")
    print("6. Train-Test comparison detects overfitting issues.")
    print("\nModel Evaluation Metrics:")
    print("- Accuracy: Overall correctness of predictions")
    print("- AUC-ROC: Ability to rank malicious domains higher than benign ones")
    print("- Precision: Of predicted malicious domains, how many are actually malicious")
    print("- Recall: Of actual malicious domains, how many we successfully detect")
    print("\nTo use this model for real-time detection, extract similar features")
    print("from domains you want to classify and use model.predict().")


if __name__ == "__main__":
    main()
