# Zomato Sentiment Analysis

    ![Last Commit](https://img.shields.io/github/last-commit/Divyansh1101/zomato-sentiment-analysis)
![CI](https://github.com/Divyansh1101/zomato-sentiment-analysis/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

    A simple NLP pipeline to classify Zomato reviews as positive/negative using Logistic Regression. If the dataset is missing, the script generates a small synthetic dataset automatically. Also produces a word cloud of frequent terms.

    ## Features
    - Data preprocessing (cleaning, tokenization, stopword removal)
- Logistic Regression classification with TF-IDF features
- Word cloud visualization saved in `results/`

    ## Requirements
    - Python 3.9+

    ## Setup & Usage
    ```bash
    pip install -r requirements.txt
    python main.py
    ```

    Outputs are saved to the `results/` folder.

    ## License
    MIT © 2025 Divyansh1101
