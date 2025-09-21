<h1>Multimodal AI for YouTube Video Success Prediction</h1>

This project explores multimodal machine learning for predicting the success of YouTube videos, operationalized as view counts. It integrates video, text, and metadata modalities with late-fusion strategies, evaluating both classification (view-range tiers) and regression (log view prediction) tasks.

📌 Overview

**Data Collection:** 500+ YouTube videos collected using the YouTube Data API.

**Preprocessing:**
<br>Frames extracted with OpenCV.
<br>Transcripts generated with Whisper ASR.
<br>Tabular features derived from metadata (e.g., subscribers, prior video stats).

**Models:**
<br>Video: Pretrained VideoMAE features + Transformer head
<br>Text: BERT-base fine-tuned with attention pooling
<br>Metadata: SVM/SVR for classification & regression

**Fusion:** Weighted late-fusion combining modality predictions (hard vote & probability averaging for classification; inverse-MSE weighting for regression).

🎯 **Results:**
<br>Classification (5 view-range classes): Achieved 48% accuracy, with metadata as the strongest modality. Fusion improved robustness over video/text alone.
<br>Regression: Explored log view prediction, but classification proved more stable and interpretable.

**Key Insight:** Metadata carries the strongest predictive power, while video and text add complementary signals.

⚙️ **Tech Stack**
<br>Python, PyTorch, scikit-learn, Optuna
<br>Hugging Face Transformers (VideoMAE, BERT, Whisper)
<br>OpenCV, yt-dlp, YouTube Data API
<br>Matplotlib/Seaborn for evaluation plots

🚀 **Future Work**
<br>Explore end-to-end video fine-tuning with VideoMAE
<br>Investigate advanced fusion (MLP/attention-based fusion)
<br>Incorporate additional modalities (audio, thumbnails, title formatting)
<br>Build a lightweight Streamlit app for creator-facing predictions

✨ This was my MSc Dissertation Project in Big Data Science. The goal was to explore how multimodal AI can support creators and platforms in anticipating content performance.
