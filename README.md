# Spam-Ham Filtering Model

A Machine Learning project that classifies email messages as Spam or Ham using TF-IDF vectorization and the Multinomial Naive Bayes algorithm.

## Overview

Email spam messages are commonly used for unwanted advertising, fraudulent offers, and misleading promotions. This project uses Natural Language Processing (NLP) and machine learning to identify whether a text message is spam or legitimate.

The model learns patterns from labelled SMS messages and predicts the category of new messages.

## Features

- Classifies SMS messages as Spam or Ham.
- Uses TF-IDF for text feature extraction.
- Implements the Multinomial Naive Bayes algorithm.
- Evaluates performance using accuracy, precision, recall, and F1-score.
- Displays a confusion matrix.
- Saves the trained model for reuse.
- Supports interactive SMS classification through the terminal.
- Displays prediction probabilities for both classes.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Project Structure

```text
spam-filter-ml/
├── spam.csv
├── train_model.py
├── spam_classifier.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── spam_model.joblib
```

The `models/` directory and its model file are generated when the training script runs.

## Dataset

This project uses the SMS Spam Collection Dataset from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains SMS messages labelled as:

- Ham: A legitimate message.
- Spam: An unwanted or potentially fraudulent message.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mahak005/spam-filter-ml.git
cd spam-ham-filtering-model
```
### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Train the Model

Run the training script:

```bash
python train_model.py
```

The script loads the dataset, prepares the messages, splits the data into training and testing sets, trains the classifier, and displays evaluation metrics.

The trained model is saved to:

```text
models/spam_model.joblib
```

## Classify a Message

Run the prediction script:

```bash
python spam_classifier.py
```

Enter an SMS message when prompted. The program displays its predicted category and the model's class probabilities.

Example:

```text
=== SMS Spam-Ham Classifier ===
Enter a message to classify.

Enter SMS: Hey, are we still meeting at 4 pm?

Message: Hey, are we still meeting at 4 pm?
Prediction: HAM
```

Type `exit` to quit the program.

*The example above illustrates the expected output format; predictions may vary depending on the trained model.*

## Model Evaluation

The model was evaluated on a held-out test set containing 1,115 messages, using an 80:20 train-test split.

| Metric | Result |
|---|---:|
| Accuracy | 96.86% |
| Ham Precision | 96.50% |
| Ham Recall | 100.00% |
| Spam Precision | 100.00% |
| Spam Recall | 76.51% |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 966 | 0 |
| Spam | 35 | 114 |

The model correctly classified 966 ham messages and 114 spam messages in the test set. It missed 35 spam messages, which indicates that improving spam recall is a potential area for future work.

## Future Improvements

- Improve spam detection recall.
- Experiment with additional machine learning algorithms.
- Perform further text preprocessing and feature engineering.
- Develop a graphical user interface.
- Deploy the model as a web application.

## Author

**Mahak**

GitHub: https://github.com/mahak005
