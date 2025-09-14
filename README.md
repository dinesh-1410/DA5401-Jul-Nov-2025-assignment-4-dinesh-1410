# DA5401 Assignment 4: GMM-Based Synthetic Sampling for Imbalanced Data

## Student Information
- **Name:** Saggurthi Dinesh
- **Roll Number:** BE21B032
- **Email:** be21b032@smail.iitm.ac.in
- **Course:** DA5401 - Data Analytics Laboratory
- **Assignment:** Assignment 4

## Project Structure
```
DA5401/A4/
├── DA5401_A4_Saggurthi_Dinesh_BE21B032.ipynb  # Main notebook
└── README.md                        # This file
```

## Requirements
- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  scipy
  ```

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Dataset
- **Source**: Credit Card Fraud Detection dataset
- **Features**: Pre-processed using PCA (V1-V28) + Time + Amount
- **Class Distribution**: 
  - Non-fraudulent: ~99.781%
  - Fraudulent: ~0.219%
- **Imbalance Ratio**: ~456.5:1

## Implementation Details

### Part A: Baseline Model and Data Analysis [10 points]
- Dataset loading and comprehensive analysis
- Class distribution visualization with multiple chart types
- Baseline Logistic Regression model on imbalanced data
- Evaluation using appropriate metrics (Precision, Recall, F1-score)
- Detailed explanation of why accuracy is misleading for imbalanced data

### Part B: GMM-Based Synthetic Sampling [35 points]

#### 1. Theoretical Foundation [5 points]
- Comprehensive comparison between GMM and SMOTE approaches
- Explanation of GMM advantages for complex data distributions
- Discussion of sub-population modeling and density-aware generation
- Analysis of covariance structure preservation

#### 2. GMM Implementation [10 points]
- Fitted Gaussian Mixture Model to minority class only
- Used AIC, BIC, and Silhouette scores for optimal component selection
- Justified choice of number of components using multiple criteria
- Implemented efficient clustering for large datasets

#### 3. Synthetic Data Generation [10 points]
- Generated synthetic samples using fitted GMM with quality control
- Explained probabilistic sampling process from mixture model
- Combined synthetic samples with original training data
- Implemented density-based filtering for sample quality

#### 4. Rebalancing with CBU [10 points]
- Clustering-based undersampling for majority class reduction
- GMM-based synthetic sampling for minority class expansion
- Created perfectly balanced dataset (1:1 ratio)
- Preserved important patterns in both classes

### Part C: Performance Evaluation and Conclusion [15 points]

#### 1. Model Training and Evaluation [5 points]
- Trained Logistic Regression on GMM-balanced dataset
- Evaluated on original imbalanced test set
- Comprehensive performance analysis

#### 2. Comparative Analysis [5 points]
- Detailed comparison table with baseline model
- Performance visualization charts
- Analysis of trade-offs between precision and recall
- Business impact assessment

#### 3. Final Recommendation [5 points]
- Evidence-based recommendations on GMM effectiveness
- Practical implementation strategies
- Future improvement suggestions

## Key Results

### Model Performance Summary
```
         Metric Baseline Model GMM-balanced Model Improvement
       Accuracy         0.9984             0.9830       -1.6%
      Precision         0.7381             0.1036      -86.0%
         Recall         0.4559             0.8824      +93.5%
       F1-Score         0.5636             0.1855      -67.1%
 True Negatives         30,845             30,337       -1.6%
False Positives             11                519    +4618.2%
False Negatives             37                  8      -78.4%
 True Positives             31                 60      +93.5%
```

### Key Findings
1. **Significant Recall Improvement**: 93.5% increase in fraud detection capability
2. **Trade-off Analysis**: Improved fraud detection at cost of increased false alarms
3. **Business Impact**: Nearly doubled true positive detection (31 → 60 cases)
4. **Reduced Missed Frauds**: Substantial decrease in false negatives (37 → 8 cases)

## Running the Notebook

1. **Data Preparation**:
   - Ensure `creditcard.csv` is in the project directory

2. **Execute Notebook**:
   - Run all cells sequentially
   - Expected runtime: 10-15 minutes 
