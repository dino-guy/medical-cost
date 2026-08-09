Medical Cost Classification using Machine Learning

A comparative study of five supervised classification algorithms applied to the Medical Cost dataset — predicting whether an individual falls into a Low or High medical cost category based on demographic and lifestyle factors.

Note: The original dataset target (charges) is continuous, making this natively a regression problem. It was converted to binary classification using a median-split threshold solely to satisfy an assignment requirement of comparing five classification algorithms. See Methodology for details.

Overview
	
Dataset	Medical Cost (5,000 records, 7 attributes)
Task	Binary classification — Low Cost (0) vs. High Cost (1)
Algorithms	Logistic Regression, Decision Tree, KNN, SVM, Naïve Bayes
Best Model	SVM — 81.20% accuracy, 80.78% F1-score
Environment	Python 3.11, Jupyter Notebook, scikit-learn
Table of Contents
Problem Statement
Dataset
Methodology
Results
Confusion Matrix
Why SVM and Logistic Regression Win
Q&A Summary
Project Structure
Getting Started
Conclusion
Problem Statement

Medical treatment cost is influenced by demographic and lifestyle factors including age, body mass index, smoking habit, number of dependent children, sex, and geographic region. The objective is to classify an individual into a Low Medical Cost or High Medical Cost category and compare five supervised classification algorithms under an identical experimental setup.

Dataset
5,000 records × 7 attributes: age, sex, bmi, children, smoker, region, charges
Target created as:
  charges ≤ median  →  cost_class = 0   (Low Cost)
  charges >  median  →  cost_class = 1   (High Cost)
charges is dropped from the input features after target creation to prevent target leakage.
Methodology
Encoding — categorical attributes (sex, smoker, region) one-hot encoded to 8 numeric features.
Train/test split — 80/20 (4,000 train / 1,000 test), stratified to preserve the near-50/50 class balance created by the median split.
Scaling — StandardScaler fit on the training set only, applied to Logistic Regression, KNN, and SVM (distance/margin-sensitive models). Decision Tree and Naïve Bayes use unscaled data.
Training & evaluation — all five models trained identically; Accuracy, Precision, Recall, F1-score, and training time recorded for each.
Results
S.No.	Algorithm	Accuracy (%)	Precision (%)	Recall (%)	F1-score (%)	Time (s)
1	Logistic Regression	81.00	81.89	79.60	80.73	0.07564
2	Decision Tree	74.00	74.00	74.00	74.00	0.02901
3	KNN	78.20	78.89	77.00	77.94	0.01751
4	SVM	81.20	82.64	79.00	80.78	0.37761
5	Naïve Bayes	69.50	100.00	39.00	56.12	0.00274

SVM achieved the highest accuracy (81.20%), narrowly ahead of Logistic Regression (81.00%). Naïve Bayes trained fastest (0.00274 s) but had the lowest accuracy (69.50%).

Confusion Matrix — Best Model (SVM)

On the 1,000-record test set:

	Predicted Low	Predicted High
Actual Low	TN = 417	FP = 83
Actual High	FN = 105	TP = 395

Errors are spread fairly evenly across both classes, so the model is not biased toward either category. In a healthcare-cost setting, the 105 false negatives are the more consequential error — genuinely high-cost individuals the model under-predicts as low-cost.

Why SVM and Logistic Regression Win

Feature-importance analysis (via Decision Tree) shows age, bmi, and smoker status account for ≈85% of the predictive signal, while sex and region contribute very little. Because charges rise in a broadly monotonic, near-linear way with these dominant variables, the Low/High boundary is close to linear — which is exactly why SVM and Logistic Regression, both smooth-global-boundary methods, come out ahead.

Decision Tree (74.0%) — axis-parallel splits over-fit the training partition without depth limits, hurting generalisation on a smooth boundary.
KNN (78.2%) — performs reasonably after scaling but struggles near the median threshold, where the two classes overlap heavily.
Naïve Bayes — the outlier: 100% precision but only 39% recall. Its conditional-independence assumption breaks because age and bmi jointly drive cost, making it label High Cost only when very confident — correct every time, but missing 61% of true High Cost cases.
Q & A Summary
Question	Answer
Highest accuracy?	SVM — 81.20%, best F1-score (80.78%)
Why?	Near-linear decision boundary fits SVM's margin maximisation well
Fastest to train?	Naïve Bayes — 0.00274 s
Best for small datasets?	SVM (support-vector dependent) & Naïve Bayes (few parameters)
Best for large datasets?	Logistic Regression — statistically tied with SVM at a fraction of the cost
Right evaluation metric?	Accuracy (balanced classes), backed by Precision/Recall/F1 and the confusion matrix
Ready for industry use?	Only as a decision-support tool, pending fairness audits, privacy review, and real-world validation
Project Structure
medical-cost/
├── Medical_Cost_Classification.ipynb   # Full analysis notebook (16-step workflow)
├── medical_cost.csv                    # Dataset (5,000 records)
├── requirements.txt                    # Python dependencies
├── Medical_Cost_Classification_Report.pdf   # 2-page written report
├── accuracy_comparison_table.csv       # Results table
├── accuracy_comparison.png             # Accuracy comparison chart
├── confusion_matrix.png                # Confusion matrix (best model)
├── training_time_comparison.png        # Training time chart (log scale)
├── feature_importance.png              # Decision Tree feature importances
└── README.md
Getting Started
Prerequisites
Python 3.9 – 3.12
Jupyter Notebook / JupyterLab
Installation
bash
git clone https://github.com/dino-guy/medical-cost.git
cd medical-cost
pip install -r requirements.txt
Usage
Open Medical_Cost_Classification.ipynb in Jupyter.
Set FILE_PATH = "medical_cost.csv" in Step 2 (already the default).
Run all cells — the notebook will:
Load and inspect the dataset
Create the binary cost_class target via median split
Encode categorical features and scale numeric ones
Train all five classifiers and record metrics
Generate the accuracy comparison table and charts
Print the confusion matrix and classification report for the best model

No GPU required — the full pipeline runs on 5,000 records in well under a second of total training time (excluding SVM's ~0.4 s).

Conclusion

Across five classifiers trained on an identical 80/20 stratified split, SVM produced the best raw accuracy (81.20%), but Logistic Regression matched it almost exactly at far lower computational cost, making it the more deployable choice at scale. Naïve Bayes trained fastest but predicted weakest, illustrating why speed and accuracy must be weighed together rather than optimised in isolation. The ~81% ceiling reflects genuine, unavoidable overlap between classes near the median threshold rather than a modelling shortfall.

Author

Dineshkumar P Reg No: 24ADR024 B.Tech – Artificial Intelligence & Data Science

License

This project was created for academic purposes as part of a Machine Learning coursework assignment.
