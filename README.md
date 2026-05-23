# Naive Bayes SMS Spam Filter

## Introduction

This project builds a simple SMS spam filter using the Naive Bayes machine learning algorithm. It demonstrates how to:

- load and inspect a labeled SMS dataset,
- clean and tokenize text data,
- train a probabilistic classifier,
- evaluate performance using recall,
- and run a complete end-to-end SMS spam detection pipeline.

The dataset contains SMS messages labeled as `spam` or `ham` (not spam). The product is designed for learning and lab practice, and it shows how a basic text classifier can be implemented from scratch.

## Project Structure

- `Lab67/naiveBayesClassifier.py`
  - Defines the `NaiveBayesFilter` class.
  - Implements training (`fit`), prediction (`predict`), probability scoring (`predict_proba`), and spam recall evaluation (`score`).
- `Lab67/spam_filter.py`
  - Loads the dataset and performs preprocessing.
  - Splits data into training and test sets.
  - Trains the Naive Bayes classifier and prints evaluation results.
- `Lab67/SMSSpamCollection.csv`
  - The dataset file with two columns: `Label` and `SMS`.
- `Lab6&7-NaiveBayes_Ky.pdf`
  - The original lab instruction document.

## What to Do in Each Problem

This repository follows a series of lab problems in a common SMS spam classification workflow:

### Problem 1: Load and inspect the dataset

- Read `Lab67/SMSSpamCollection.csv` using pandas.
- Check the dataset size and first few rows.
- Confirm that labels are either `spam` or `ham`.

### Problem 2: Clean and tokenize SMS text

- Remove punctuation and non-word characters.
- Convert all text to lowercase.
- Split each message into a list of tokens (words).
- Keep the processed messages in `X_train` and `X_test` for model input.

### Problem 3: Implement Naive Bayes training

- Build the vocabulary from training data.
- Compute prior probabilities for `spam` and `ham`.
- Count word frequencies separately for spam and ham messages.
- Apply Laplace smoothing to avoid zero probabilities.

### Problem 4: Predict and score results

- Compute class probabilities for each message.
- Predict `spam` if the spam probability is higher than ham.
- Measure performance using recall, especially spam recall.
- Compare results on both training and test datasets.

### Problem 5: Run the full pipeline

- Execute `Lab67/spam_filter.py`.
- Confirm the dataset loads correctly even when running from a different working directory.
- Review printed output for dataset statistics and recall scores.

## Requirements

- Python 3.13+ (or any compatible Python 3.x installation)
- pandas
- numpy

## Installation

From the project root, install dependencies with:

```bash
pip install pandas numpy
```

## How to Run

Execute the script from the repository root:

```bash
python Lab67/spam_filter.py
```

The script uses the dataset file path relative to `spam_filter.py`, so it works correctly from anywhere.

## Algorithm

This project uses the Multinomial Naive Bayes algorithm:

1. Preprocess SMS text by removing punctuation and converting text to lowercase.
2. Tokenize each message into words.
3. Build a vocabulary from the training set.
4. Compute prior probabilities for `spam` and `ham`.
5. Count word frequencies in `spam` and `ham` messages.
6. Apply Laplace smoothing to handle unseen words.
7. Use log probabilities for numeric stability.
8. Predict labels by comparing spam and ham likelihoods.

## Result

The script prints dataset size, training/test split details, and recall scores.

Example output from a successful run:

- `recall train of NB: 0.9666666666666667`
- `recall test of NB: 0.9659863945578231`

<img width="820" height="576" alt="image" src="https://github.com/user-attachments/assets/f14e6e3a-8609-4bac-b44b-d6b0f4862618" />

This means the model correctly identified about 96.6% of spam messages in both train and test sets.

## Notes

- The model uses Laplace smoothing for unseen words.
- Log probabilities improve numeric stability during prediction.
- `spam_filter.py` locates the dataset based on its own file location.
