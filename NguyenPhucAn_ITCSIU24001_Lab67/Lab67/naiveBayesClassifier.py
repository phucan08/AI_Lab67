import pandas as pd
import numpy as np

class NaiveBayesFilter:
    def __init__(self):
        self.data = []
        self.vocabulary = []  # returns tuple of unique words
        self.p_spam = 0  # Probability of Spam
        self.p_ham = 0  # Probability of Ham
        self.parameters_spam = {}
        self.parameters_ham = {}

    def fit(self, X, y):
        # X is expected to be an iterable of token lists, y contains labels 'spam' or 'ham'
        self.data = list(zip(X, y))

        # Build vocabulary
        self.vocabulary = sorted({word for message in X for word in message})

        # Prior probabilities
        total_messages = len(y)
        spam_messages = sum(1 for label in y if label == 'spam')
        ham_messages = total_messages - spam_messages
        self.p_spam = spam_messages / total_messages if total_messages > 0 else 0
        self.p_ham = ham_messages / total_messages if total_messages > 0 else 0

        # Initialize word counts with Laplace smoothing
        self.parameters_spam = {word: 1 for word in self.vocabulary}
        self.parameters_ham = {word: 1 for word in self.vocabulary}
        self.total_spam_words = len(self.vocabulary)
        self.total_ham_words = len(self.vocabulary)

        for message, label in self.data:
            if label == 'spam':
                for word in message:
                    if word in self.parameters_spam:
                        self.parameters_spam[word] += 1
                        self.total_spam_words += 1
            else:
                for word in message:
                    if word in self.parameters_ham:
                        self.parameters_ham[word] += 1
                        self.total_ham_words += 1

        return self.data

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return ['spam' if prob >= 0.5 else 'ham' for prob in probabilities]

    def predict_proba(self, X):
        proba = []
        for message in X:
            # Use log probabilities for numeric stability
            log_spam = np.log(self.p_spam) if self.p_spam > 0 else -np.inf
            log_ham = np.log(self.p_ham) if self.p_ham > 0 else -np.inf

            for word in message:
                if word in self.vocabulary:
                    log_spam += np.log(self.parameters_spam[word] / self.total_spam_words)
                    log_ham += np.log(self.parameters_ham[word] / self.total_ham_words)
                else:
                    # Unknown words get the same smoothing probability
                    log_spam += np.log(1 / self.total_spam_words)
                    log_ham += np.log(1 / self.total_ham_words)

            # Convert log odds to probability
            max_log = max(log_spam, log_ham)
            exp_spam = np.exp(log_spam - max_log)
            exp_ham = np.exp(log_ham - max_log)
            proba.append(exp_spam / (exp_spam + exp_ham))

        return proba

    def score(self, true_labels, predict_labels):
        # Return spam recall: true spam correctly identified / all actual spam
        if len(true_labels) == 0:
            return 0
        true_spam = 0
        true_positive_spam = 0
        for expected, actual in zip(true_labels, predict_labels):
            if expected == 'spam':
                true_spam += 1
                if actual == 'spam':
                    true_positive_spam += 1
        return true_positive_spam / true_spam if true_spam > 0 else 0

