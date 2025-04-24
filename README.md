# 🧠 ADL Project: Prescient Vision-Language Agents for Dynamic Manipulation

This repository contains the codebase for our final project submission to **10-707: Advanced Deep Learning** at **Carnegie Mellon University (Spring 2025)**. Our work explores the integration of **scene prediction models** with **Vision-Language Agents (VLAs)** to enable anticipatory reasoning in dynamic robotic tasks—specifically targeting **nonprehensile object manipulation**.

## 🔍 Project Summary

We develop a predictive VLA pipeline where future visual states are inferred using models like **SwinLSTM**, allowing the agent to reason not just over observed inputs, but also over anticipated future scenes. This setup proves crucial in tasks where temporal precision and proactive planning are required.

Our results demonstrate that **prescient agents**—which incorporate future frame prediction—significantly outperform **reactive VLAs**, even under constraints of delayed inference.

## 🧪 Highlights

- **Future-aware VLA planning** via frame prediction with SwinLSTM.
- **Custom simulated dataset** based on robotic rollouts with moving objects.
- **Quantitative and qualitative evaluation** using SSIM, PSNR, MSE, and task success.
- **Exploration of integration trade-offs** (inference time vs. success rate).

## ⚙️ Setup & Usage

### Requirements

- Python 3.9
- PyTorch ≥ 1.13
- OpenCV, Matplotlib, tqdm, numpy, pyyaml

### Install Dependencies

```bash
git clone https://github.com/yatharthahuja/adl-project.git
cd adl-project
pip install -r requirements.txt
```

### Training SwinLSTM

```bash
python train.py
```

### Evaluating the Model

```bash
python test.py
```

## 📊 Results

- **Fine-tuned SwinLSTM** outperformed pretrained (zero-shot) models by nearly **100x** in prediction accuracy (measured by MSE).
- **Prescient VLAs** consistently succeeded in tasks where **vanilla VLAs** failed due to delayed response or lack of foresight.
- Performance degraded under extreme conditions (e.g., fast object motion), suggesting future work on prediction horizon control.

## 🧾 Acknowledgment

This project was developed as part of the coursework for **10-707: Advanced Deep Learning**, Spring 2025, Carnegie Mellon University.
