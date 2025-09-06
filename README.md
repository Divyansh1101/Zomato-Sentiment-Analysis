
# 🍽️ Zomato Sentiment Analysis

A **Natural Language Processing (NLP)** project that classifies Zomato restaurant reviews into **positive** or **negative** sentiments using **Logistic Regression**. If the real dataset is unavailable, the script automatically generates a synthetic dataset to demonstrate the workflow.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Last Commit](https://img.shields.io/github/last-commit/Divyansh1101/Zomato-Sentiment-Analysis?style=flat-square)  

---

## ✨ Features

- 🧠 **Sentiment Classification** – Classify restaurant reviews into positive or negative sentiments.  
- 🧪 **Data Preprocessing** – Clean, tokenize, and vectorize text data for better model performance.  
- 🌐 **Word Cloud Visualization** – Visualize frequent words from reviews for both positive and negative sentiments.  
- 🔄 **Synthetic Data Generation** – Automatically create sample data if the real dataset is not found.  
- 📊 **Model Evaluation** – Display accuracy, confusion matrix, and classification report for the sentiment classifier.  
- ⚙ **Modular Design** – Easy to extend with other classification algorithms or datasets.

---

## 📂 Project Structure

```
Zomato-Sentiment-Analysis/
├── main.py                # Main script for sentiment analysis and visualization
├── data/                  # Dataset folder (optional, if using real data)
├── requirements.txt       # Python dependencies
├── LICENSE                # Project license file
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Divyansh1101/Zomato-Sentiment-Analysis.git
cd Zomato-Sentiment-Analysis
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Analysis  
```bash
python main.py
```
The script will load the dataset from the `data/` folder if available, or generate synthetic data otherwise. It will then perform sentiment analysis, evaluate the model, and create word cloud visualizations.

---

## 📂 Dataset Format (Optional)

You can use your own dataset placed in the `data/` folder. The dataset should be a CSV file with at least the following columns:

```csv
review_text,sentiment
"Great food and service!",positive
"Terrible experience, food was cold.",negative
...
```

If the dataset is missing, the script will automatically create a synthetic dataset for demonstration purposes.

---

## 📊 Example Output

- ✅ **Accuracy Score** – Displays how well the model performed.  
- 📈 **Confusion Matrix** – Visual representation of true vs predicted sentiments.  
- 🌐 **Word Clouds** – Separate visualizations for words commonly found in positive and negative reviews.

*(Include screenshots or sample outputs here for clarity.)*

---

## ⚙️ Requirements

- Python **3.8+**  
- Libraries listed in `requirements.txt`:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `wordcloud`

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) for providing powerful machine learning algorithms  
- [WordCloud](https://github.com/amueller/word_cloud) for creating visually appealing word clouds  
- [Matplotlib](https://matplotlib.org/) for helping visualize data  
- [Pandas](https://pandas.pydata.org/) for efficient data handling  
- Open-source resources and contributors who made data science more accessible
