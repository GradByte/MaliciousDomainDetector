# Malicious Domain Detection using Machine Learning

A machine learning system that classifies domains as benign or malicious using Random Forest classification on DNS logs and domain characteristics.

![Model Results](https://github.com/GradByte/MaliciousDomainDetector/blob/main/malicious_domain_detection_results.png)

## 🎯 Overview

This project implements a comprehensive malicious domain detection system that analyzes **33 different features** extracted from domain names, DNS records, and threat intelligence data to identify potentially harmful websites. The model achieves high accuracy in distinguishing between legitimate and malicious domains, making it suitable for:

- **Cybersecurity applications**: Real-time threat detection
- **Email security**: Phishing URL identification  
- **Network monitoring**: DNS traffic analysis
- **Browser protection**: Warning systems for suspicious sites

## 📊 Dataset

**Source**: [Mendeley Data - Benign and Malicious Domains Dataset](https://data.mendeley.com/datasets/623sshkdrz/5)

The dataset contains DNS logs with labeled benign (0) and malicious (1) domains, featuring:
- Domain structure characteristics
- DNS record information
- Email authentication data (SPF, DKIM, DMARC)
- Reputation scores
- Registration metadata
- Statistical features (entropy, character ratios, sequences)

## 🔍 Features Used (33 Total)

### 1. Entropy-Based Features
- **Entropy**: Measures randomness in domain names (DGA detection)
- **EntropyOfSubDomains**: Subdomain randomness analysis

### 2. Character Composition
- **ConsonantRatio**: Vowel/consonant balance
- **NumericRatio**: Numeric character percentage
- **SpecialCharRatio**: Special character percentage
- **VowelRatio**: Vowel percentage
- **StrangeCharacters**: Unusual character count

### 3. Sequence Features
- **ConsonantSequence**: Longest consonant sequence
- **VowelSequence**: Longest vowel sequence
- **NumericSequence**: Longest numeric sequence
- **SpecialCharSequence**: Longest special character sequence

### 4. Domain Structure
- **DomainLength**: Total domain length
- **SubdomainNumber**: Count of subdomains
- **TLD**: Top-level domain (.com, .tk, etc.)

### 5. DNS & Email Security
- **MXDnsResponse**: Mail exchange record presence
- **TXTDnsResponse**: TXT record presence
- **HasSPFInfo**: SPF record exists
- **HasDkimInfo**: DKIM record exists
- **HasDmarcInfo**: DMARC record exists

### 6. Infrastructure & Reputation
- **DomainInAlexaDB**: Presence in popular domain database
- **CommonPorts**: Standard port usage
- **HttpResponseCode**: HTTP response status
- **IpReputation**: IP threat intelligence score
- **DomainReputation**: Domain threat intelligence score

### 7. Registration Data
- **CreationDate**: Domain age indicator
- **LastUpdateDate**: Recent modification indicator
- **ASN**: Autonomous System Number
- **CountryCode**: Registration country
- **DNSRecordType**: DNS record classification

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
pip (Python package manager)
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### Download Dataset
1. Download the dataset from [Mendeley](https://data.mendeley.com/datasets/623sshkdrz/5)
2. Place `BenignAndMaliciousDataset.csv` in a `datasets/` folder

## 💻 Usage

### Basic Usage
```bash
python malicious_domain_detector.py
```

### What It Does
The script automatically:
1. ✅ Loads and preprocesses the dataset
2. ✅ Engineers 33 features for classification
3. ✅ Trains a Random Forest model (80/20 train-test split)
4. ✅ Evaluates performance with multiple metrics
5. ✅ Generates visualization of results
6. ✅ Performs overfitting analysis
7. ✅ Runs 5-fold cross-validation

### Output Files
- `malicious_domain_detection_results.png` - Comprehensive visualization with 4 plots

## 📈 Model Performance

### Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Overall correctness | % of correct predictions |
| **AUC-ROC** | Ranking ability | How well it separates classes (0.5-1.0) |
| **Precision** | Malicious prediction accuracy | Of flagged domains, % truly malicious |
| **Recall** | Detection rate | Of actual malicious domains, % detected |
| **F1-Score** | Precision-recall balance | Harmonic mean of precision & recall |

### Overfitting Detection
The model includes automated overfitting analysis:
- ✓ **Good**: Accuracy gap < 5%, AUC gap < 5%
- ⚠ **Moderate**: Gaps between 5-10%
- ✗ **Overfitting**: Gaps > 10%

### Cross-Validation
5-fold cross-validation ensures:
- Consistent performance across different data splits
- Model generalizes well to unseen data
- Robustness against training set bias

## 🔬 How It Works

### 1. Data Preprocessing
```python
# Convert boolean features to binary (0/1)
# Handle missing values with median imputation
# Encode categorical features (TLD, Country Code, etc.)
```

### 2. Model Training
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split node
    class_weight='balanced' # Handle class imbalance
)
```

**Why Random Forest?**
- Handles mixed feature types (numeric, boolean, categorical)
- Resistant to overfitting through ensemble learning
- Provides feature importance rankings
- Robust to outliers and missing values
- No need for feature scaling

### 3. Feature Importance Analysis
The model identifies which characteristics are most predictive of malicious behavior:
- Top features typically include: Entropy, Domain Reputation, IP Reputation
- Helps security analysts understand attack patterns
- Enables feature engineering for future improvements

### 4. Prediction Pipeline
```python
# Example: Classify a new domain
new_domain_features = extract_features(domain)
prediction = model.predict(new_domain_features)
confidence = model.predict_proba(new_domain_features)

# Output: "MALICIOUS" (85% confidence) or "BENIGN" (92% confidence)
```

## 📊 Visualizations

The results image includes 4 key plots:

### 1. Top 15 Feature Importance
- Bar chart showing which features matter most
- Helps interpret model decisions
- Guides security policy development

### 2. Confusion Matrix
- Visual breakdown of predictions
- Shows True Positives, False Positives, etc.
- Identifies error patterns

### 3. ROC Curve
- Trade-off between detection rate and false alarms
- Compares train vs test performance
- AUC score quantifies classification quality

### 4. Train vs Test Comparison
- Side-by-side accuracy and AUC comparison
- Visual overfitting detection
- Ensures model reliability

## 🛡️ Real-World Applications

### Security Operations Centers (SOC)
```python
# Monitor DNS traffic in real-time
for domain in dns_logs:
    features = extract_features(domain)
    if model.predict(features) == 1:
        alert_security_team(domain)
```

### Email Security Gateways
```python
# Scan URLs in emails
for url in email.links:
    domain = extract_domain(url)
    if is_malicious(domain):
        quarantine_email()
```

### Browser Extensions
```python
# Warn users before visiting suspicious sites
def on_url_change(url):
    if model.predict_proba(url)[1] > 0.7:
        show_warning("This site may be malicious")
```

## 📝 Example Predictions

```python
# High-risk indicators
Domain: "xj3k2m9pqr8s.tk"
Features:
  - High entropy: 4.8
  - Long domain: 120 chars
  - Strange characters: 15
  - No SPF/DKIM/DMARC
  - Not in Alexa DB
  - Recent creation date
  
Prediction: MALICIOUS (94% confidence)

# Low-risk indicators  
Domain: "amazon.com"
Features:
  - Low entropy: 2.1
  - Normal length: 10 chars
  - In Alexa top 100
  - Has SPF/DKIM/DMARC
  - Established domain
  
Prediction: BENIGN (99% confidence)
```

## 🔧 Customization

### Adjust Model Parameters
```python
# Increase trees for better accuracy (slower)
RandomForestClassifier(n_estimators=200)

# Reduce depth to prevent overfitting
RandomForestClassifier(max_depth=15)

# Adjust class weights for imbalanced data
RandomForestClassifier(class_weight={0: 1, 1: 3})
```

### Add Custom Features
```python
def extract_custom_features(domain):
    features['has_https'] = check_ssl_certificate(domain)
    features['domain_age_days'] = calculate_age(domain)
    features['typosquatting_score'] = check_similarity_to_popular_domains(domain)
    return features
```

## 📚 Understanding the Metrics

### Confusion Matrix Breakdown
```
                    Predicted
                Benign      Malicious
Actual Benign     TN           FP       ← False Positive = False alarm
     Malicious    FN           TP       ← False Negative = Missed threat!
```

- **True Negative (TN)**: Correctly identified safe domain ✓
- **False Positive (FP)**: Safe domain flagged as malicious (annoying but safe)
- **False Negative (FN)**: Malicious domain missed (DANGEROUS!)
- **True Positive (TP)**: Correctly caught malicious domain ✓

### Which Metric Matters Most?

**For security applications**: Prioritize **Recall** (catch all threats)
- Better to have false alarms than miss real threats
- Example: 95% recall = we catch 95% of malicious domains

**For user experience**: Balance **Precision** and **Recall**
- Too many false positives → users ignore warnings
- F1-Score provides the optimal balance

## ⚠️ Limitations & Considerations

### Current Limitations
1. **Feature Dependency**: Requires access to DNS records and reputation databases
2. **Adversarial Attacks**: Attackers who know the features could craft evasive domains
3. **Concept Drift**: Attack patterns evolve; model needs periodic retraining
4. **False Positives**: Legitimate new domains may be flagged
5. **Dataset Bias**: Performance depends on training data representation

### Best Practices
- ✅ Retrain model monthly with new threat intelligence
- ✅ Combine with other security layers (signature-based, behavioral)
- ✅ Implement human review for high-stakes decisions
- ✅ Monitor false positive rates in production
- ✅ Use confidence thresholds (e.g., only alert if >80% confidence)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add deep learning models (LSTM, CNN for domain names)
- Implement real-time feature extraction from live DNS
- Expand dataset with recent threat intelligence
- Add explainability features (SHAP, LIME)
- Create API endpoint for production deployment

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Dataset**: Mendeley Data - Benign and Malicious Domains
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn
- **Inspiration**: DNS-based threat detection research

## 📞 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Submit a pull request with improvements

---

**⭐ If you find this project useful, please consider giving it a star!**

## 🔗 References

1. [Mendeley Dataset](https://data.mendeley.com/datasets/623sshkdrz/5)
2. [Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
3. [Domain Generation Algorithms (DGA)](https://en.wikipedia.org/wiki/Domain_generation_algorithm)
4. [DNS Security Best Practices](https://www.cloudflare.com/learning/dns/dns-security/)

---

**Last Updated**: February 2026
