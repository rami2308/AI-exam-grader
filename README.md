# AI Exam Grader — Automated Handwritten Exam Grading System

An end-to-end AI pipeline that allows lecturers to upload handwritten exam papers and receive structured, justified grades automatically — consistently, at scale, and in a fraction of the time manual grading takes.

## The Problem

Grading exams manually is slow and inconsistent. For large classes, reviewing dozens of handwritten papers becomes a bottleneck that consumes hours of a lecturer's time. Beyond speed, subjectivity is a deeper issue — the same answer can receive different scores depending on the grader's fatigue or interpretation. Students deserve fair, standardised evaluation.

## How It Works

1. Handwritten exam papers are scanned and loaded from disk
2. **Qwen2.5-VL-7B-Instruct** (4-bit quantized) reads each handwritten answer using chain-of-thought prompting
3. The model evaluates the answer against a rubric — scoring for **logic** and **accuracy**
4. **GraderNet** (custom neural network) maps those two scores into a final grade (0–100)
5. Results are returned in a structured, reviewable format per student

## Architecture

```
Scanned Exam Paper (question + solution images)
              ↓
Qwen2.5-VL-7B-Instruct — 4-bit quantized via BitsAndBytesConfig
  Chain-of-thought prompting: evaluates logic + accuracy
              ↓
     [logic_score, acc_score]
              ↓
GraderNet — nn.Linear(2, 1) → final grade
              ↓
     Structured output per student
```

## Tech Stack

| Layer | Technology |
|---|---|
| Vision-Language Model | Qwen2.5-VL-7B-Instruct |
| Quantization | BitsAndBytesConfig (4-bit NF4) |
| Model Fine-Tuning | SFT + GRPO via ms-swift |
| Inference Serving | vLLM |
| Grading Network | Custom GraderNet (PyTorch) |
| Image Processing | OpenCV, Pillow |
| Feature Extraction | Custom pipeline (logic + accuracy scores) |
| UI | Gradio |
| Backend | Python |

## Project Structure

```
AI-exam-grader/
├── universal_grader.py        # Main grader: loads Qwen, runs chain-of-thought evaluation
├── interactive_grader.py      # GraderNet inference: scores → final grade
├── math_grader.py             # Math-specific grading logic
├── text_grader.py             # Text-specific grading logic
├── train_model.py             # GraderNet training pipeline
├── extract_features.py        # Feature extraction from model output
├── generate_synthetic_data.py # Synthetic training data generation
├── grader_model.pth           # Trained GraderNet weights
├── scaler.pkl                 # Feature scaler for GraderNet input
└── src/
    ├── evpm/                  # Expression-aware Visual Prompting Module
    └── instr_tuning/          # SFT + GRPO training scripts and prompts
        └── scripts/
            ├── sft.sh         # Stage 1: Supervised fine-tuning
            ├── grpo.sh        # Stage 2: GRPO reinforcement learning
            ├── serve_vllm.sh  # Deploy model with vLLM
            └── serve_rewards.sh
```

## Status

| Component | Status |
|---|---|
| Qwen2.5-VL-7B inference pipeline | ✅ Complete |
| GraderNet (logic + accuracy → grade) | ✅ Complete |
| SFT fine-tuning | ✅ Complete |
| GRPO reinforcement learning | ✅ Complete |
| Math & text subject graders | ✅ Complete |
| Synthetic data generation | ✅ Complete |
| Gradio UI | 🔄 In progress |
| Rubric input interface | 🔄 Planned |

## Base Research

This project builds on the VEHME architecture:
> *VEHME: Vision-Language Model For Evaluating Handwritten Mathematics Expressions* — EMNLP 2025

## Author

**Rami Tal** — 3rd Year Computer Science, SCE Ashdod
Final Year Project — Supervised by SCE Department of Computer Science
[LinkedIn](https://linkedin.com/in/rami-tal) · [GitHub](https://github.com/rami2308)
