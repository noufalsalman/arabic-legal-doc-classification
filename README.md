# Legal Arabic Document Classification

An end-to-end NLP pipeline for classifying Saudi commercial court cases: predicting case verdicts and legal topics from Arabic case facts, built on the ALARB dataset.

**Course project, CS476: Natural Language Processing**

## Overview

Arabic legal text is a genuinely hard NLP problem: formal Modern Standard Arabic register, long documents (avg. 181 words), morphologically rich vocabulary, and domain-specific tokens like Hijri dates, monetary figures, and legal citations. This project builds a full pipeline, from raw court case text to a working interactive demo, tackling two tasks on 13,341 Saudi commercial court cases:

- **Task A, Verdict Prediction:** classify a case into one of 5 outcomes (Fully Accepted, Rejected, Settlement, Dismissed, Partially Accepted)
- **Task B, Legal Topic Classification:** classify the legal domain of a dispute (Sales & Supply Contracts, General Commercial Disputes, Commercial Partnerships)

## Results

| Model | Task | Accuracy | F1 (weighted) |
|---|---|---|---|
| **LinearSVC + TF-IDF (bigram)** ★ | A | **71.6%** | **0.707** |
| **LinearSVC + TF-IDF (bigram)** ★ | B | **69.6%** | **0.691** |
| Logistic Regression + TF-IDF | A | 69.4% | 0.672 |
| Feedforward ANN + TF-IDF | A | 69.1% | 0.680 |
| Feedforward ANN + TF-IDF | B | 66.6% | 0.573 (class collapse) |
| Logistic Regression + BoW | A | 66.2% | 0.659 |
| Logistic Regression + AraVec embeddings | A | 57.9% | 0.424 |

**Best model:** LinearSVC with TF-IDF bigrams, grid-searched (C=1.0), for both tasks.

## Key Findings

1. **TF-IDF consistently beats Bag-of-Words and embeddings.** Domain vocabulary matters more than raw frequency.
2. **AraVec (Twitter-trained) is a poor fit for legal text.** Near-total vocabulary mismatch (100% OOV in our run) versus formal legal register.
3. **Stemming gives a modest boost** (~0.6 pts F1). Bigram TF-IDF already captures much of the morphological signal.
4. **Class imbalance is the central challenge.** "Partially Accepted" (30 of 9,374 examples) was never predicted correctly by any model.
5. **The ANN overfits quickly** on sparse TF-IDF vectors and collapsed to majority-class prediction on Task B. A sequence-aware model (BiLSTM, AraBERT) would likely do better.

## Approach

**Preprocessing (7-step Arabic-specific pipeline):** character normalization, diacritics removal, legal token handling (dates/amounts/case numbers replaced with placeholders), punctuation/non-Arabic removal, stopword removal (including legal-domain stopwords), ISRIStemmer stemming (74.5% vocabulary reduction), whitespace normalization.

**Feature extraction:** Bag-of-Words, TF-IDF (unigram and bigram), word/character n-grams, AraVec embeddings (as a comparison baseline).

**Models:** Logistic Regression, LinearSVC (best), Feedforward ANN, plus an implemented-but-not-executed AraBERT fine-tuning path (toggleable via `RUN_BERT`, requires GPU).

**Experiments:** 5 controlled comparisons isolating the effect of BoW vs. TF-IDF, stemming, word vs. character n-grams, embeddings vs. TF-IDF, and model architecture.

## Demo

An interactive Gradio demo accepts free-text Arabic case facts and returns both a verdict prediction and a legal topic prediction with confidence scores, using calibrated LinearSVC models (`CalibratedClassifierCV`).

## Ethical Considerations

The report includes a full ethical impact assessment covering prediction bias toward majority classes, risks of over-reliance on a ~71%-accuracy tool in real legal contexts, litigant privacy, and responsible-deployment guidelines (human review for minority classes, confidence reporting, anonymization). See the full report for details.

## Tech Stack

Python, scikit-learn, TensorFlow/Keras, pyarabic, NLTK (ISRIStemmer), Gradio

## Project Structure

notebook/arabic_legal_doc_classification.ipynb
report/arabic-legal-doc-classification.pdf
presentation/arabic-legal-doc-classification-slides.pdf
README.md
