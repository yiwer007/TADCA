# TADCA

**Official implementation of "Seeing Beyond Masks: Temporal-Aware Differential Cross-Modal Attention for Multimodal Sentiment Analysis"**

## 📖 Introduction

**TADCA** (Temporal-Aware Differential Cross-Modal Attention) is a robust framework designed for Multimodal Sentiment Analysis (MSA). By leveraging a differential attention mechanism and temporal-aware modeling, TADCA effectively captures subtle emotional shifts across vision, audio, and text modalities while maintaining high robustness against noise and masks.

## 📊 Datasets

The model is evaluated on four benchmark datasets. You can download them via the following links:

- **CMU-MOSI & CMU-MOSEI**: Access via [CMU-MultimodalDataSDK](https://github.com/Jie-Xie/CMU-MultimodalDataSDK).
- **UR-FUNNY**: Access via [UR-FUNNY Repository](https://github.com/ROC-HCI/UR-FUNNY).
- **CH-SIMS**: For the Chinese multimodal dataset, refer to the [ACL Anthology paper](https://aclanthology.org/2020.acl-main.343.pdf).

## 🚀 Training

To train the model on a specific dataset (e.g., MOSI), run:

Bash

```
python main.py --dataset mosi
```

## 🛠️ Environment Requirements

- **Python**: 3.8.8
- **PyTorch**: 1.8.1
- **NumPy**: 1.20.0