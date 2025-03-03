# SMS Spam Classification Project

## Project Overview
With the increasing use of mobile messaging, spam messages have become a major issue, leading to fraud, scams, and unnecessary clutter in users’ inboxes. This project aims to develop an effective machine learning model that can automatically classify SMS messages as **spam** or **ham** (legitimate). By comparing various supervised and unsupervised learning models, we aim to build a high-accuracy spam detection system that minimizes false negatives (missed spam) and false positives (misclassified ham messages).

## Goals of the Project
- Develop a spam detection model that efficiently classifies SMS messages.
- Compare multiple machine learning approaches, including supervised and unsupervised learning methods.
- Enhance feature engineering for improved classification performance.
- Evaluate model performance using accuracy, precision, recall, and F1-score.

## Technologies Used
- **Programming**: Python  
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Natural Language Processing: `nltk`, `scikit-learn`, `TfidfVectorizer`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - Machine Learning:
    - Supervised: `Logistic Regression`, `SVM`, `Random Forest`, `XGBoost`
    - Unsupervised: `K-Means`, `PCA`, `Hierarchical Clustering`, `LDA`, `DBSCAN`

## Dataset
- **Source**: [UCI Machine Learning Repository - SMS Spam Collection](https://doi.org/10.24432/C5CC84)  
- **Size**: 5,574 SMS messages labeled as **ham** or **spam**.
- **Preprocessing**:
  - Text cleaning (lowercasing, punctuation removal, tokenization, stopword removal).
  - TF-IDF vectorization for feature extraction.

## Exploratory Data Analysis (EDA)
- **Class Distribution**: 
  - Ham: 4,827 messages
  - Spam: 747 messages  
  - The dataset is imbalanced, with more ham messages than spam.
- **Message Length Analysis**:
  - Spam messages tend to be longer than ham messages.
- **Wordclouds & N-Gram Analysis**:
  - Spam messages frequently contain words like `"FREE"`, `"win"`, `"claim"`, `"prize"`.
  - Ham messages contain more conversational words like `"ok"`, `"time"`, `"love"`, `"see"`.
- **Stopword Ratio**:
  - Ham messages contain a higher proportion of stopwords, indicating a more natural conversation style.
  - Spam messages use fewer stopwords, making them more keyword-driven.

## Model Performance Summary

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|--------|---------|---------------|------------|-------------|
| Logistic Regression | 95% | 0.95 | 0.71 | 0.81 |
| Random Forest | 97% | 1.00 | 0.82 | 0.90 |
| **SVM (Best Model)** | **98%** | **0.97** | **0.88** | **0.92** |
| XGBoost | 97% | 0.96 | 0.85 | 0.90 |

SVM (Support Vector Machine) performed the best, achieving the highest recall and accuracy in spam detection.

## Challenges in Unsupervised Learning
- K-Means and PCA failed to form clear clusters due to the complexity of SMS text data.
- Hierarchical Clustering and DBSCAN produced poor silhouette scores, indicating weak separation.
- LDA and NMF (Topic Modeling) provided some insights but did not distinctly separate spam and ham messages.

## Conclusion
- Supervised learning outperforms unsupervised approaches for SMS spam classification.  
- SVM with tuned parameters is the best-performing model.  
- Handling class imbalance (e.g., using SMOTE or weighted loss functions) can further improve spam detection.

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install scikit-learn nltk matplotlib seaborn xgboost wordcloud

