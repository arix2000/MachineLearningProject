# import wykorzystywanych bibliotek
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# importowanie csv do ramki danych
patients = pd.read_csv('data/cancer_patient_data_sets.csv')
print(patients.head())
print("Data imported correctly.")

# analiza danych statystycznych
print(patients.describe())
print(patients.info())

print("\nWhether there are null values in the data set?")
print(patients.isnull().values.any())

# rozkład płci
gender_dist = patients['Gender'].hist(bins=2, label=["Female", "Male"])
title = 'GENDER DISTRIBUTION'
gender_dist.set_title(title)
plt.gca().set_xticklabels(['Male', 'Female'])
plt.xticks(np.arange(min(patients['Gender']), max(patients['Gender']) + 1, 1))
plt.show()

# ryzyko wystąpienia raka płuc
cancer_risk = patients['Level'].hist(bins=3)
title = 'RISK OF DEVELOPING LUNG CANCER'
gender_dist.set_title(title)
plt.show()
