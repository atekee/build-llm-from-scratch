---
title: Usta Model Chat
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎓 Mastering Large Language Models: Build Your Own LLM from Scratch

Run first version of the model in colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1itWjR18elzBIKawGvc1cMpBzQu6i1ZmU?usp=sharing)

## 🧩 Course Subtitle: **Master the Architecture, Pretraining, and Fine-Tuning of Transformer-Based AI Systems**

---

- Huggingface transformers repository
- Attention is All You Need and GPT2/GPT3 papers
- https://bbycroft.net/llm
- https://github.com/malibayram
- https://www.comet.com/site/blog/explainable-ai-for-transformers/
- https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing#scrollTo=TG-dQt3NOlub
- https://jalammar.github.io/illustrated-transformer/
- https://www.topbots.com/deconstructing-bert-part-1/
- https://jalammar.github.io/illustrated-bert/
- https://huggingface.co/google/gemma-3-4b-it#inputs-and-outputs
- https://huggingface.co/spaces/alibayram/turkish_tiktokenizer
- https://tiktokenizer.vercel.app/
- https://github.com/indri-voice/audiotoken

---

- https://youtu.be/QWNxQIq0hMo?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms
- https://youtu.be/Xpr8D6LeAtw?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu
- https://youtu.be/yAcWnfsZhzo?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11
- https://youtu.be/ZLbVdvOoTKM
- https://youtu.be/UU1WVnMk4E8
- https://youtu.be/tFHeUSJAYbE?list=PLz-ep5RbHosU2hnz5ejezwaYpdMutMVB0
- https://youtu.be/UPtG_38Oq8o?list=PLs8w1Cdi-zvalz9ltXmarqyeQ49wfKFqf
- https://youtu.be/ZLbVdvOoTKM

## 📘 Course Description

Dive deep into the world of Large Language Models (LLMs) and go far beyond simply calling APIs! In this hands-on, full-stack course, you'll build a working **GPT-style (decoder-only) Transformer model** from scratch using Python and PyTorch. We’ll dissect every single component—from understanding **tokens**, constructing **embedding layers**, integrating **positional encodings**, implementing the revolutionary **self-attention mechanism**, to assembling **complete Transformer blocks**.

You’ll learn the theory behind pretraining LLMs on massive corpora (e.g., WikiText), then **fine-tune them via Supervised Fine-Tuning (SFT)** for a specific task such as Question-Answering. By the end of this journey, not only will you deeply understand how models like ChatGPT work, but you'll have hands-on experience in **building, training, and adapting your own model**. This course is designed for developers, data scientists, and AI practitioners who want to go beyond using prebuilt tools and become **engineers capable of constructing LLMs from the ground up**.

---

## 👥 Target Audience

- Developers, data scientists, ML engineers, and AI enthusiasts with Python and deep learning basics.
- Anyone curious about the internals of LLMs.
- Professionals aiming to gain deep LLM knowledge to stand out in interviews or research.

---

## 🔧 Prerequisites

- Solid Python knowledge (functions, classes, data structures)
- Familiarity with PyTorch (`nn.Module`, tensors, autograd, optimizers)
- Basic understanding of NumPy and data manipulation
- **GPU access** (e.g., Colab, Kaggle, or local CUDA setup) highly recommended

---

## 🚀 Learning Outcomes

By the end of this course, you will:

- Understand and code a **decoder-only GPT-style Transformer architecture**
- Create and apply **tokenization** strategies (e.g., BPE) and use tools like `tiktoken`
- Implement **token embeddings (`nn.Embedding`)** and understand their role in semantic representation
- Grasp and implement **positional encodings**
- Implement the **self-attention mechanism** (Q, K, V vectors and scaled dot-product attention)
- Apply **causal attention masks** for language modeling
- Build **multi-head attention** to capture diverse contextual signals
- Train LLMs for **next-token prediction using CrossEntropyLoss**
- Perform **Supervised Fine-Tuning (SFT)** for tasks like Q&A
- Learn from foundational papers such as **"Attention Is All You Need"** and the **GPT series**
- Optimize training with batching, GPU usage, and hyperparameter tuning
- Save/load models and monitor training progress
- Evaluate model performance with key metrics

---

## 📚 Curriculum (Each lecture ~25 mins)

---

### 📦 Module 0: Introduction & Environment Setup

- **Lesson 0.1 — Welcome to the LLM Revolution**

  - Course goals, what we will build
  - Why learn LLMs from scratch?
  - Open-source vs closed-source models (GPT-4 vs LLaMA 3)

- **Lesson 0.2 — Core Concepts: Autoregression, Transformers, Pretraining vs Fine-tuning**

- **Lesson 0.3 — Setting Up Your Deep Learning Environment**

  - Python, PyTorch, `datasets`, `tiktoken`, `transformers`
  - GPU on Colab / Kaggle

---

### 📦 Module 1: Data — The Fuel for LLMs

- **Lesson 1.1 — Understanding Text & The Role of Tokenization**

  - Words, subwords, characters
  - BPE explained

- **Lesson 1.2 — Practical Tokenization with `tiktoken`**

  - Encoding/decoding tokens
  - Vocabulary size, special tokens

- **Lesson 1.3 — Exploring Pretraining Datasets (WikiText / OpenWebText)**

- **Lesson 1.4 — Preparing Inputs & Targets**

  - Creating (input, target) pairs for next-token prediction
  - Managing `block_size`, `batch_size`

- **Lesson 1.5 — Efficient Data Handling with PyTorch `DataLoader`**
  - Custom Dataset and batching logic

---

### 📦 Module 2: Representing Meaning — Embeddings & Positional Awareness

- **Lesson 2.1 — From Tokens to Vectors: The Magic of Embeddings**

  - One-hot vs learned embeddings
  - Distributional semantics (Word2Vec intro)

- **Lesson 2.2 — Implementing Token Embeddings**

  - Using `nn.Embedding`, shape walkthrough

- **Lesson 2.3 — The Problem of Order: Why Position Matters**

- **Lesson 2.4 — Implementing & Visualizing Positional Encodings**
  - Sinusoidal encoding
  - Adding PE to token embeddings

---

### 📦 Module 3: The Attention Engine Room

- **Lesson 3.0 — PyTorch Quick Refresher**

  - Tensors, `nn.Module`, `autograd`, optimizers
  - Simple MLP example

- **Lesson 3.1 — Self-Attention: The Heart of Transformers**

  - Query, Key, Value intuition

- **Lesson 3.2 — Scaled Dot-Product Attention Calculation**

  - Softmax, dot products, matrix math

- **Lesson 3.3 — Decoder-Only Masking for Causal LM**

  - Preventing information leakage

- **Lesson 3.4 — Multi-Head Attention: Capturing Diverse Context**
  - Why multiple heads matter
  - Building MHA from scratch

---

### 📦 Module 4: Building the Transformer Block

- **Lesson 4.1 — LayerNorm & Residual Connections**

  - Improving stability & training

- **Lesson 4.2 — Feed-Forward Networks in Transformers**

  - 2-layer projection logic

- **Lesson 4.3 — Full Decoder Block Assembly**
  - MHA → Add & Norm → FFN → Add & Norm

---

### 📦 Module 5: Assembling & Pretraining Our GPT

- **Lesson 5.1 — Stacking Decoder Blocks & Output Head**

  - Final linear projection to vocab size

- **Lesson 5.2 — Objective: Next Token Prediction & Loss Function**

- **Lesson 5.3 — Optimizer Setup & Learning Rate Scheduler**

  - AdamW, cosine schedule with warmup

- **Lesson 5.4 — Pretraining Loop Pt. 1: Forward + Backward**

- **Lesson 5.5 — Pretraining Loop Pt. 2: Gradient Clipping & Logging**

- **Lesson 5.6 — Running Pretraining & Monitoring Loss Curves**

- **Lesson 5.7 — Inference with Your Trained Model**
  - Greedy decoding & sampling strategies

---

### 📦 Module 6: Specialization — Supervised Fine-Tuning (SFT) for Q&A

- **Lesson 6.1 — Why Fine-Tuning? Pretraining Isn’t Enough**

- **Lesson 6.2 — Intro to Supervised Fine-Tuning**

- **Lesson 6.3 — Q&A Dataset (e.g., GammaCorpus / Turkish QA)**

- **Lesson 6.4 — Adapting the Training Loop for SFT**

- **Lesson 6.5 — Running Fine-Tuning for QA Task**

- **Lesson 6.6 — Inference with Fine-Tuned QA Model**

---

### 📦 Module 7: Reflections, Advanced Topics, and Next Steps

- **Lesson 7.1 — Scaling Laws, Memory, and Compute Challenges**

- **Lesson 7.2 — Advanced Topics Overview (for Self-Study)**

  - RLHF, LoRA, quantization, Mixture of Ex

- **Lesson 7.3 — Course Summary & Your LLM Journey Ahead**

---

### 📦 Module 8: Projects & Evaluation

- **Lesson 8.1 — Final Project: Build and Evaluate Your Own GPT Model**

- **Lesson 8.2 — Final Project Demo: Turkish Domain-Specific LLM**

- **Lesson 8.3 — Bonus: Evaluate Tokenizer Quality & MMLU Correlation**

- **Lesson 8.4 — Final Exam: MCQs and Coding Questions**

- **Lesson 8.5 — Publishing Your Model to Hugging Face**

---

## 🎁 Course Deliverables

- ✅ 25+ Video Lectures (~25 minutes each)
- 🧪 Colab Notebooks per module
- 📝 Quizzes & Assignments
- 🎓 Final Capstone Project
- 📁 Source Code Templates
- 🧾 Certificate of Completion

---

## 📎 Notes

- Based on academic and open-source research, including:
  - GPT papers (1–3)
  - Transformer ("Attention Is All You Need")
  - Hugging Face & nanoGPT tools

---
