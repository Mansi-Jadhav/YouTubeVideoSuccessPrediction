<h1>Multimodal AI for YouTube Video Success Prediction</h1>

This project explores multimodal machine learning for predicting the success of YouTube videos, operationalized as view counts. It integrates video, text, and metadata modalities with late-fusion strategies, evaluating both classification (view-range tiers) and regression (log view prediction) tasks.

📌 Overview

**Data Collection:** 500+ YouTube videos collected using the YouTube Data API.

**Preprocessing:**
Frames extracted with OpenCV.
Transcripts generated with Whisper ASR.
Tabular features derived from metadata (e.g., subscribers, prior video stats).

**Models:**
Video: Pretrained VideoMAE features + Transformer head
Text: BERT-base fine-tuned with attention pooling
Metadata: SVM/SVR for classification & regression

**Fusion:** Weighted late-fusion combining modality predictions (hard vote & probability averaging for classification; inverse-MSE weighting for regression).

🎯 **Results:**
Classification (5 view-range classes): Achieved 48% accuracy, with metadata as the strongest modality. Fusion improved robustness over video/text alone.
Regression: Explored log view prediction, but classification proved more stable and interpretable.

**Key Insight:* Metadata carries the strongest predictive power, while video and text add complementary signals.

⚙️ **Tech Stack**
Python, PyTorch, scikit-learn, Optuna
Hugging Face Transformers (VideoMAE, BERT, Whisper)
OpenCV, yt-dlp, YouTube Data API
Matplotlib/Seaborn for evaluation plots

🚀 **Future Work**
Explore end-to-end video fine-tuning with VideoMAE
Investigate advanced fusion (MLP/attention-based fusion)
Incorporate additional modalities (audio, thumbnails, title formatting)
Build a lightweight Streamlit app for creator-facing predictions

✨ This was my MSc Dissertation Project in Big Data Science. The goal was to explore how multimodal AI can support creators and platforms in anticipating content performance.
