# FashionVision: Multi-Architecture CNN Analysis

A comprehensive deep learning project that implements and evaluates multiple Convolutional Neural Network (CNN) architectures for fashion image classification using the Fashion MNIST dataset.

## Project Overview

This project focuses on:
- Training and evaluating 10 different CNN model architectures
- Feature extraction and dimensionality reduction using intermediate layer representations
- Clustering analysis and visualization techniques
- Mystery label identification using extracted features

## Dataset

The project uses the **Fashion MNIST** dataset, which contains:
- Training data: Fashion item images with corresponding labels
- Test data: Fashion item images for evaluation
- 5 different fashion categories (based on the code showing `num_classes=5`)

## Model Architectures

The project implements and evaluates 10 different CNN models (Mod 1 through Mod 10) with varying architectures and hyperparameters. Each model is trained and evaluated to find the best performing architecture.

### Model Performance Results

Based on the code analysis, the model accuracies achieved are:
- **Mod 1**: 87.62%
- **Mod 2**: 85.85%
- **Mod 3**: 30.00%
- **Mod 4**: 88.95%
- **Mod 5**: 92.31%
- **Mod 6**: 93.30%
- **Mod 7**: 93.86%
- **Mod 8**: 92.96%
- **Mod 9**: 92.03%
- **Mod 10**: 86.16%

**Best performing model: Mod 7 with 93.86% accuracy**

## Technical Implementation

### Core Technologies
- **TensorFlow/Keras**: Deep learning framework for CNN implementation
- **Scikit-learn**: Machine learning utilities for clustering and dimensionality reduction
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **VisualKeras**: Model architecture visualization

### Key Features

#### 1. CNN Model Training
- Multiple CNN architectures with different layer configurations
- Early stopping implementation to prevent overfitting
- Comprehensive model evaluation and comparison
- Training time and testing time tracking

#### 2. Feature Extraction
- Intermediate layer feature extraction from trained models
- Dimensionality reduction using Principal Component Analysis (PCA)
- Feature encoding for downstream analysis

#### 3. Clustering Analysis
- **DBSCAN Clustering**: Density-based clustering with eps=2, min_samples=30
- **K-means Clustering**: K-means with 5 clusters
- Visualization of clustering results using PCA-reduced features

#### 4. Dimensionality Reduction & Visualization
- **PCA**: Principal component analysis for feature reduction
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for 2D visualization
- **Isomap**: Isometric mapping for non-linear dimensionality reduction

#### 5. Mystery Label Identification
- SVM classifier implementation using extracted features
- Classification of unlabeled fashion items
- Accuracy evaluation on test data

## Project Structure

```
Fashion_CNN_Model/
├── CNN_Fashionmnist.ipynb    # Main Jupyter notebook with complete implementation
├── README.md                 # Project documentation
└── report.pdf               # Detailed project report
```

## Key Findings

1. **Model Performance**: The best performing model achieved 93.86% accuracy, significantly outperforming the baseline models
2. **Feature Learning**: Intermediate layer representations effectively capture discriminative features for fashion classification
3. **Clustering Insights**: Both DBSCAN and K-means clustering reveal meaningful patterns in the learned feature space
4. **Dimensionality Reduction**: Multiple visualization techniques (PCA, t-SNE, Isomap) provide complementary insights into data structure

## Usage

1. Open `CNN_Fashionmnist.ipynb` in Jupyter Notebook or Google Colab
2. Ensure all required dependencies are installed
3. Run the notebook cells sequentially to reproduce the results
4. The notebook includes comprehensive documentation and visualization of results

## Dependencies

- TensorFlow 2.x
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- VisualKeras

## Results Summary

This project demonstrates the effectiveness of CNN architectures for fashion image classification, achieving up to 93.86% accuracy. The comprehensive analysis includes model comparison, feature extraction, clustering analysis, and multiple visualization techniques, providing valuable insights into both the classification performance and the learned feature representations.
