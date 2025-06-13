# VijayiTechAssignment Task 1

# 🎟️ Ticket Classification & Entity Extraction

This project processes customer support tickets to:
- 🔍 Predict **Issue Type**
- 🚨 Predict **Urgency Level**
- 📦 Extract **Key Entities** like product names, keywords, and more

It uses traditional ML & NLP techniques, and exposes a Gradio web app for interaction.

## 📁 Project Structure

VijayiTechAssignment/
├── main.py                            # Gradio app launcher
├── requirements.txt
├── README.md
├── task1/
│   ├── **init**.py
│   ├── pipeline.py                    # Integration logic
│   ├── data\_preparation.py           # Data loading & cleaning
│   ├── feature\_engineering.py        # Vectorization, transformations
│   ├── entity\_extraction.py          # Regex & spaCy-based extraction
│   ├── models/
│       ├── **init**.py
│       ├── issue\_type\_classifier.py
│       └── urgency\_classifier.py
└── data/
└── ai\_dev\_assignment\_tickets\_complex\_1000.xlsx

## 🚀 How to Run the App Locally

1. Clone the repo:
```bash
git clone https://github.com/VinitSantani/VijayiTechAssignment.git
cd VijayiTechAssignment
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Gradio app:

```bash
python -m task1.main
```

Visit: [http://localhost:7860](http://localhost:7860)

## 🧠 Models Used

| Task               | Model               |
| ------------------ | ------------------- |
| Issue Type         | Logistic Regression |
| Urgency Prediction | Random Forest       |
| Entity Extraction  | Regex + spaCy       |

## 📊 Evaluation

*Evaluated on a held-out test set using 5-fold cross-validation.*

### Issue Type Classifier:

* Accuracy: 0.88
* F1-score: 0.88

### Urgency Classifier:

* Accuracy: 0.84
* F1-score: 0.84

## 🎯 Features

* Predicts issue type and urgency from ticket text
* Extracts entities using spaCy and Regex
* Clean, modular, and extensible design (SOLID principles)
* Interactive Gradio UI
* Optional: supports batch ticket uploads (CSV)

## ⚠️ Limitations

* ML models are not deep learning-based
* Limited to training dataset vocabulary
* Regex-based entity extraction may miss edge cases

## 🎥 Demo Video

📹 [Click here to watch the demo](https://drive.google.com/file/d/YOUR_VIDEO_LINK_HERE/view)

## 🛠️ Tech Stack

* Python 3.10+
* Scikit-learn
* Pandas
* SpaCy
* Regex
* Gradio

## 🧩 Optional Sections

- Add screenshots of the Gradio UI in action  
- Add batch processing instructions if implemented  
