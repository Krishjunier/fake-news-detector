# fake-news-detector

A simple **BERT-based** fake news detection tool that classifies news text as **REAL** or **FAKE** with confidence scores.  
It uses a fine-tuned BERT model and matches input against known news snippets for faster, more accurate detection.

---

## Features

- Pre-trained BERT model for fake news classification
- Checks input text against predefined real/fake news examples
- Predicts label and confidence score for unseen news texts
- Easy to use Python function interface

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fake-news-detector.git
    cd fake-news-detector
    ```

2. Install dependencies:
    ```bash
    pip install transformers torch
    ```

3. Place your fine-tuned BERT model folder named `fake-news-bert-model` inside the project root.

---
