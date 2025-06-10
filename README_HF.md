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

# Usta Model Chat 🤖

A chat interface for a custom transformer language model built from scratch! This model specializes in geographical knowledge including countries, capitals, and cities.

## Model Details

- **Architecture**: Custom Transformer-based Language Model
- **Vocabulary Size**: 64 tokens
- **Embedding Dimension**: 12
- **Attention Heads**: 4
- **Layers**: 8
- **Context Length**: 32 tokens

## Features

- Interactive chat interface
- Real-time text generation
- Focused on geographical knowledge
- Built from scratch implementation

## Usage

Simply type your message in the chat box and the model will generate a response based on its training on geographical data.

### Example Prompts

- "the capital of france"
- "tell me about spain"
- "what is the capital of united states"
- "paris is in"
- "germany and its capital"

## Technical Implementation

This model was implemented from scratch using PyTorch, featuring:

- Custom tokenizer for geographical vocabulary
- Multi-head self-attention mechanism
- Layer normalization
- MLP feed-forward networks
- Causal (autoregressive) text generation

## Limitations

- Limited vocabulary (64 tokens)
- Focused domain (geography)
- Small model size (experimental/educational purposes)

## Code

The complete implementation is available in the repository, including:

- Custom model architecture (`UstaModel`)
- Custom tokenizer (`UstaTokenizer`)
- Training notebooks and data processing
- Gradio web interface

Built with ❤️ using PyTorch and Gradio.
