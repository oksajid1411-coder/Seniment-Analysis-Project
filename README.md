# 💭 Sentiment Analyzer

A Machine Learning-based **Sentiment Analysis web application** built with Python and Streamlit. The application takes a text input and predicts one of four sentiment classes — **Positive, Negative, Neutral, or Irrelevant** — along with the model's confidence and probability breakdown.

## ✨ Features

- 📝 Text input for sentiment analysis
- 🤖 Machine Learning-based sentiment prediction
- 🎯 Confidence score for the predicted class
- 📊 Probability breakdown for all sentiment classes
- 🔤 Displays character and word counts
- 📋 Detailed probability table
- 🎨 Modern, responsive Streamlit interface
- ✨ Animated UI elements and glass-style cards
- ♿ Reduced-motion support in the interface

The application uses a trained vectorizer and Logistic Regression model loaded from `.joblib` files. The model's stored class order is used when mapping predictions to display labels, helping avoid class-order mismatches. 

## 🧠 Sentiment Classes

The application supports four classes:

| Class | Meaning |
|---|---|
| 😊 Positive | Positive sentiment |
| 😢 Negative | Negative sentiment |
| 😐 Neutral | Neutral sentiment |
| 🤔 Irrelevant | Text that is not relevant to the target sentiment |

For integer-encoded labels, the application uses:

```text
0 → Irrelevant
1 → Negative
2 → Neutral
3 → Positive
```

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- **CountVectorizer / TF-IDF Vectorization**
- **Logistic Regression**
- HTML/CSS for custom UI styling

## 📂 Project Structure

```text
Sentiment-Analyzer/
│
├── app.py
├── project_2.ipynb
├── count_vec.joblib
├── model_lr.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

> `count_vec.joblib` and `model_lr.joblib` are required by the Streamlit application.

## 🔄 How It Works

```text
User enters text
       ↓
Text is vectorized
       ↓
Trained Logistic Regression model
       ↓
Prediction + probability scores
       ↓
Sentiment + confidence displayed
```

The Streamlit application transforms the input using the saved vectorizer and then uses the saved model to generate the prediction and class probabilities.

## 🧪 Model Development

The accompanying notebook performs the main Machine Learning workflow:

1. Load the Twitter training dataset.
2. Inspect the dataset and its columns.
3. Rename/select relevant text and sentiment fields.
4. Clean the text using a custom `clean_text()` function.
5. Remove duplicate records.
6. Encode the sentiment labels using `OrdinalEncoder`.
7. Split the data into training and testing sets.
8. Create **TF-IDF** and **CountVectorizer** representations.
9. Train multiple models, including:
   - K-Nearest Neighbors
   - Logistic Regression
   - Random Forest
   - An experimental neural-network model
10. Compare model performance using `classification_report`.
11. Select **Logistic Regression with TF-IDF vectorization** as the chosen model.
12. Save the trained model/vectorizer for use in the Streamlit application.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start Streamlit with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 📦 Required Model Files

The application expects these files in the same project directory as `app.py`:

```text
count_vec.joblib
model_lr.joblib
```

If either file is missing, the application will stop and display an error explaining that the model files could not be loaded.

## 📊 Application Output

After entering text and clicking **Analyze Sentiment**, the application displays:

- Predicted sentiment
- Confidence percentage
- Character count
- Word count
- Probability for each sentiment class
- Sorted detailed probability breakdown
- The original input text

## 🔐 Notes

- Do not commit API keys, passwords, authentication tokens, `.env` files, virtual environments, or other sensitive files to GitHub.
- Keep generated model files in the repository only if their size and licensing allow it.
- If the model files are too large for normal Git usage, consider Git LFS or another model-storage solution.

## 📄 License

Add your preferred license here before publishing the repository.

For example, if you choose the MIT License, include a `LICENSE` file containing the official MIT License text.

## 👤 Author

**Your Name**

Built with ❤️ using Python, Streamlit & Machine Learning.
