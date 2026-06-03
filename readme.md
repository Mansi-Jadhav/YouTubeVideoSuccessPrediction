<h1># Multimodal AI for YouTube Video Success Prediction</h1>

A multimodal machine learning framework for predicting YouTube video performance using video content, textual information, and channel metadata. This project was developed as part of my MSc Dissertation in Big Data Science and investigates how different modalities contribute to forecasting video popularity.

## Project Overview

Predicting the success of online content remains a challenging problem due to the complex interactions between visual, textual, and contextual factors. This project explores both regression and classification approaches for estimating YouTube video performance, measured primarily through view counts.

The study evaluates the predictive power of multiple modalities individually and in combination, providing insights into which factors are most influential in determining video success.

### Dataset Creation

A custom dataset of over 500 YouTube videos was collected using the YouTube Data API across diverse categories and channel sizes.

The dataset includes:

* Video content (sampled video frames)
* Video transcripts generated using automatic speech recognition
* Channel and engagement metadata
* Historical channel performance statistics
* Video-level popularity metrics

### Feature Extraction

**Video Modality**

* Frame extraction using OpenCV
* Visual feature extraction using VideoMAE
* Temporal modelling with Transformer architectures

**Text Modality**

* Video transcripts generated using Whisper ASR
* Text embeddings obtained using BERT-base
* Fine-tuning with attention-based pooling for view count prediction

**Metadata Modality**

* Channel statistics (subscriber count, upload history)
* Historical engagement metrics
* Video category and duration information
* Structured feature engineering and preprocessing

### Models Evaluated

| Modality | Models                                            |
| -------- | ------------------------------------------------- |
| Video    | VideoMAE + Transformer                            |
| Text     | Fine-tuned BERT                                   |
| Metadata | Random Forest, SVM, SVR, XGBoost, Neural Networks |
| Fusion   | Late Fusion (Weighted Averaging & Voting)         |

### Multimodal Fusion

Predictions from individual modalities were combined using late-fusion strategies:

* Hard Voting (Classification)
* Probability Averaging
* Weighted Regression Fusion
* Inverse-MSE Weighting

This enabled the system to leverage complementary information from each modality while preserving modality-specific strengths.

## Results

### Classification (5 View-Count Categories)

* Achieved **48% accuracy** across five view-count classes
* Metadata emerged as the strongest standalone predictor
* Fusion models improved robustness and overall consistency
* Video and text modalities provided complementary signals that enhanced prediction quality

### Regression (Log View Count Prediction)

* Evaluated continuous view-count prediction using log-transformed targets
* Compared multiple machine learning and deep learning approaches
* Classification-based modelling proved more stable and interpretable for practical applications

## Key Findings

* **Channel metadata is the most predictive modality** for estimating video success.
* **Visual and textual content provide complementary information** that can improve multimodal performance.
* **Late-fusion approaches outperform several single-modality baselines**, demonstrating the value of multimodal learning.
* Historical channel performance is a stronger predictor than content features alone.

## Technology Stack

**Languages & Frameworks**

* Python
* PyTorch
* Scikit-learn
* Optuna

**Deep Learning Models**

* VideoMAE
* BERT
* Whisper

**Data Collection & Processing**

* YouTube Data API
* OpenCV
* yt-dlp

**Visualization & Analysis**

* Matplotlib
* Seaborn

## Future Improvements

* End-to-end VideoMAE fine-tuning
* Attention-based multimodal fusion networks
* Thumbnail and audio feature integration
* Explainable AI techniques for feature importance analysis
* Streamlit-based deployment for creator-facing predictions

## Research Impact

This work demonstrates how multimodal AI can be applied to understand and predict online content performance. The findings provide insights into the relative importance of content features versus channel-level signals and establish a foundation for future research in multimodal popularity prediction systems.
