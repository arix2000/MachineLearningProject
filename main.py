# import wykorzystywanych bibliotek
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# importowanie csv do ramki danych
patients = pd.read_csv('data/cancer_patient_data_sets.csv')
patients=patients.replace(to_replace="Low", value=1)
patients=patients.replace(to_replace="Medium", value=2)
patients=patients.replace(to_replace="High", value=3)
print(patients.head())
print("Data imported correctly.")

# analiza danych statystycznych
print(patients.describe())
print(patients.info())

print("\nWhether there are null values in the data set?")
print(patients.isnull().values.any())

# rozkład płci
gender_dist = patients['Gender'].hist(bins=2, label=["Female", "Male"])
plt.xlabel("gender")
plt.ylabel("number of people")
title = 'GENDER DISTRIBUTION'
gender_dist.set_title(title)
plt.gca().set_xticklabels(['Male', 'Female'])
plt.xticks(np.arange(min(patients['Gender']), max(patients['Gender']) + 1, 1))
plt.show()

# palacze w grupie badawczej
cancer_risk = patients['Smoking'].hist(bins=8)
title = 'SMOKERS IN THE RESEARCH GROUP'
plt.xlabel("level of smoking")
plt.ylabel("number of people")
cancer_risk.set_title(title)
plt.show()

# rozkład danych z uwzglednieniem wieku
age_dist = patients['Age'].plot.box()
title = 'DATA DISTRIBUTION (BASED ON AGE)'
age_dist.set_title(title)
plt.show()

# usuwanie nieprzydatnych danych
patients.drop(['index', 'Patient Id'], axis=1, inplace = True)
print(patients.head())
print('Unneccesary data deleted!')

# wyznaczenie macierzy korelacji

correlations = patients.corr()
fig, ax = plt.subplots(figsize=(24, 24))

colormap = sns.color_palette("BrBG", 10)

sns.heatmap(correlations,
    cmap=colormap,
    annot=True,
    fmt=".2f")
plt.show()


