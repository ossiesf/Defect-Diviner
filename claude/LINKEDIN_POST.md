# LinkedIn Post Draft

---

**I tried to predict bugs with ML. Here's what actually worked (and what didn't).**

Started with what seemed obvious: extract code complexity metrics, train a classifier, predict buggy files. Simple.

**Reality check:** The model performed at 53% accuracy. Basically a coin flip.

Why? Bug fixes are often 1-2 line changes. File-level metrics barely move. Buggy code and clean code looked identical to the model.

**The fix:** I added historical features:
- How many developers touched this file?
- How many past bugs in this file?
- How much code churn?

**Results jumped to 75% ROC AUC.**

The top predictor? Past bugs. Files that had bugs before get more bugs. (Turns out this is well-documented in research - "defect density persistence")

**Lesson:** The right features beat the right algorithm every time.

Tech: PyDriller, radon, XGBoost

Code: [GitHub link]

#MachineLearning #SoftwareEngineering #DataScience #Python

---

*[Feel free to edit/personalize this - add your own voice and any specific details about your background or goals]*
