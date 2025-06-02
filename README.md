# Credit Risk Clustering

This project explores whether unsupervised learning techniques can enhance the performance of supervised credit risk classification models. By combining clustering methods with logistic regression, the project evaluates if unsupervised pretraining can improve accuracy and reduce the cost of misclassifications for credit card applicants.

## 📊 Project Overview

Financial institutions face a critical challenge: expanding their customer base while minimizing the risk of lending to bad customers. In this project, we aim to address the following management problem:

> Can unsupervised learning methods, such as clustering, improve credit risk prediction models based on customer financial and demographic data?

We compare traditional logistic regression (baseline) with models that incorporate unsupervised clustering features to evaluate improvements in classification accuracy and cost savings.

## 🔍 Dataset

The dataset used is the [German Credit Data](https://www.openml.org/d/31) from OpenML, containing:
- **1,000** credit applicants
- **20** explanatory variables (numerical and categorical)
- A binary target variable indicating credit risk (`good` or `bad`).

The data dictionary is available in `Data-Dictionary-for-German-Credit-Case.txt` for reference.

## 🧰 Methods

1. **Data Preparation**
   - One-hot encoded categorical variables.
   - Applied Min-Max normalization for consistent feature scaling.
   - Ensured no data leakage: clustering methods were trained without access to class labels.

2. **Baseline Model: Logistic Regression**
   - Trained on the full dataset using K-Fold cross-validation.
   - Evaluated using precision, recall, F1-score, and cost-sensitive metrics based on misclassification costs.

3. **Unsupervised Learning Techniques**
   - **Gaussian Mixture Model (GMM)**: Selected optimal clusters using BIC/AIC.
   - **K-Means Clustering**: Determined cluster count using Elbow Method.
   - **Hierarchical Clustering**: Dendrogram analysis for cluster count; evaluated with silhouette score.
   - Cluster labels were added as new features for logistic regression models.

4. **Evaluation**
   - Compared models using precision, recall, F1-score, and average misclassification costs.
   - Visualized distributions, clustering structures (e.g., dendrograms, BIC/AIC plots), and t-SNE projections for analysis.

## 📈 Key Findings

- The baseline logistic regression achieved:
  - **F1 Score**: 0.825
  - **Precision**: 0.703
  - **Recall**: 0.999
  - **Average Misclassification Cost**: 60.20 per fold

- Incorporating unsupervised clustering features (GMM, K-Means, Hierarchical) did **not** meaningfully improve the model's performance. Clustering structures were weak, as indicated by low silhouette scores.

- Conclusion: The dataset lacks strong clustering structures that could enhance prediction performance. Alternative methods like feature engineering and ensemble learning (e.g., Random Forest, XGBoost) are recommended for further exploration.
