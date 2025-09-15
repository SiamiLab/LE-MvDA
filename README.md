# Locally Enhanced Multi-view Discriminant Analysis (LE-MvDA)

**Dimension reduction method for multi-view data**

## 📋 Overview

LE-MvDA is a novel supervised dimensionality reduction method for multi-view classification. Unlike traditional MvDA that relies on global class separation, LE-MvDA introduces a **unified mathematical framework** that simultaneously optimizes both discriminative learning and local structure preservation within a single generalized eigenvalue problem.

## 🔧 Pipeline

![Multi-view Robot Activity Classification Pipeline](Images/Architecture.png)

**Figure:** Multi-view Robot Activity Classification Pipeline.  
The framework consists of three main stages:
1. **Feature extraction**: Pose estimation extracts keypoints from videos captured by three synchronized cameras observing robotic actions. The extracted features are converted into view-specific *time-series data*, each encoding the motion dynamics from a different viewpoint.
2. **Transformation**: These time-series sequences are mapped from their original feature spaces into a *shared subspace* using linear transformation matrices T₁, T₂, and T₃, one per view.
3. **Classification**: Performed in the common subspace, where samples from different views are projected close together based on class similarity. Shapes indicate different views (circle, square, triangle), and colors represent distinct activity classes (0–9).

## 📦 Dataset

We use the **[RoboMNIST dataset](https://github.com/SiamiLab/RoboMNIST)** — a multimodal dataset for multi-robot activity recognition (MRAR) that integrates **WiFi Channel State Information (CSI)**, **video**, and **audio** data.

The dataset was collected using two Frank
