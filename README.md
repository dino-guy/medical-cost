# 🏥 Medical Cost Classification

Predicting whether a person's medical cost will be **Low** or **High**, using
five different machine learning models — and comparing which one works best.

---

## 📌 What is this project about?

Hospitals and insurance companies want to know: *will this person's medical
bill be expensive or not?*

This project uses real-world factors — age, weight (BMI), smoking habit,
number of kids, gender, and where the person lives — to predict that answer.

The original data gives an exact cost in rupees/dollars (a number, like
₹15,000 or ₹40,000). But for this project, we simplified it into just two
categories:

- 🟢 **Low Cost** — cheaper than the average person
- 🔴 **High Cost** — more expensive than the average person

We then trained **5 different algorithms** and compared which one guesses
correctly most often.

---

## 🧠 The 5 Algorithms We Compared

| Algorithm | In Simple Words |
|---|---|
| **Logistic Regression** | Draws a straight line to separate "Low" from "High" |
| **Decision Tree** | Asks yes/no questions step by step, like a flowchart |
| **KNN** | Looks at the closest similar people and copies their answer |
| **SVM** | Draws the *best possible* line with the widest gap between groups |
| **Naïve Bayes** | Uses probability and guesses based on patterns |

---

## 🏆 Which One Won?

| Algorithm | Accuracy | Speed |
|---|---|---|
| Logistic Regression | 81.00% | Fast |
| Decision Tree | 74.00% | Fast |
| KNN | 78.20% | Fast |
| **SVM** ⭐ | **81.20%** | Slow |
| Naïve Bayes | 69.50% | ⚡ Fastest |

**SVM won on accuracy** — it got things right 81.2% of the time.
But **Logistic Regression came extremely close (81.0%) and was 5x faster to
train**, so in real life, Logistic Regression is the smarter practical choice.

![Accuracy Comparison](images/accuracy_comparison.png)

---

## 🎯 How Good Were the Predictions?

Out of 1,000 test people, here's how SVM (the best model) did:

![Confusion Matrix](images/confusion_matrix.png)

In plain terms:
- ✅ **417 people** correctly predicted as Low Cost
- ✅ **395 people** correctly predicted as High Cost
- ❌ **83 people** were actually Low Cost but predicted High Cost
- ❌ **105 people** were actually High Cost but predicted Low Cost (the
  riskier mistake — these are expensive patients the model missed)

---

## 🔍 Why Did SVM and Logistic Regression Win?

We checked which factors mattered most for predicting cost:

![Feature Importance](images/feature_importance.png)

Turns out, **just 3 things** — age, BMI, and whether someone smokes —
explain about **85% of the prediction**. Gender and region barely matter.

Because these top 3 factors affect cost in a smooth, predictable way, models
that draw a smooth line (like SVM and Logistic Regression) naturally do
better than models that make blocky step-by-step decisions (like Decision
Tree).

---

## ⏱️ Speed vs. Accuracy

![Training Time](images/training_time_comparison.png)

- **Naïve Bayes** is the fastest (almost instant) but least accurate.
- **SVM** is the slowest (still under half a second) and most accurate.
- **Logistic Regression** is a great middle ground — nearly as accurate as
  SVM, but much faster.

---

## ❓ Common Questions

**Q: Which model should I actually use?**
Logistic Regression — it's basically as accurate as SVM but way faster and
simpler.

**Q: Can this be used in a real hospital?**
Not directly. It's a good *starting point*, but before real use it needs more
testing, fairness checks (to make sure it's not biased by gender/region), and
privacy safeguards since it's health data.

**Q: Why convert a number (cost) into just Low/High?**
This project specifically required comparing classification algorithms, so
the continuous cost value was split into two simple categories using the
midpoint (median) of all costs.

---

## 📁 What's in This Repo

```
medical-cost/
├── Medical_Cost_Classification.ipynb   → the code (open in Jupyter)
├── medical_cost.csv                    → the dataset (5,000 people)
├── requirements.txt                    → what to install before running
├── images/                             → all charts used in this README
└── README.md                           → this file
```

---

## 🚀 How to Run It Yourself

```bash
# 1. Download this project
git clone https://github.com/dino-guy/medical-cost.git
cd medical-cost

# 2. Install what's needed
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Medical_Cost_Classification.ipynb

# 4. Click "Run All" — that's it!
```

No fancy computer needed — this runs fine on a normal laptop, no GPU
required. The whole thing trains in under a second.

---


*This project was built as part of a Machine Learning coursework assignment.*
