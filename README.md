# Legal Arabic Document Classification

Classification of Arabic legal documents using classical NLP preprocessing and a comparison of traditional and deep learning models, built on the ALARB dataset.

## Overview

This project tackles the challenge of automatically categorizing Arabic legal text — a domain with unique preprocessing needs (diacritics, morphological complexity) and comparatively limited NLP tooling relative to English. The pipeline covers the full path from raw legal text to a working classification demo.

## Approach

**Preprocessing**
- Text normalization
- Diacritics removal
- Stemming

**Feature Extraction**
- Bag-of-Words (BoW)
- TF-IDF
- N-grams
- AraVec (pretrained Arabic word embeddings)

**Models Compared**
- Logistic Regression
- Linear SVC
- Feedforward ANN
- AraBERT (fine-tuned, as an extension beyond classical methods)

## Results

*(Add your best accuracy/F1 numbers here per model — this is the first thing recruiters will look at.)*

## Demo

Includes an interactive Gradio demo for classifying new legal text samples in real time.

## Tech Stack

Python · scikit-learn · TensorFlow/PyTorch (for ANN/AraBERT) · Gradio · AraVec

## Project Structure

\```
├── data/                  # ALARB dataset (or link/instructions if not redistributable)
├── notebooks/
│   └── alarb_classification.ipynb
├── demo/
│   └── gradio_app.py
├── README.md
\```

## How to Run

\```bash
pip install -r requirements.txt
python demo/gradio_app.py
\```

## Dataset

[ALARB (Arabic Legal Text Classification dataset)](link-if-public) — link to the source/paper if publicly citable.
