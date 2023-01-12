# import wykorzystywanych bibliotek
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importowanie csv do ramki danych
patients = pd.read_csv('data/cancer_patient_data_sets.csv')
patients = patients.replace(to_replace="Low", value=0)
patients = patients.replace(to_replace="Medium", value=0)
patients = patients.replace(to_replace="High", value=1)
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
cancer_risk = patients['Smoking'].hist(bins=7)
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
patients.drop(
    ['index', 'Patient Id', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity',
     'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing',
     'Swallowing Difficulty',
     'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough'], axis=1, inplace=True)
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

# uzupełnienie pustych wartości
patients.fillna(patients.mean(), inplace=True)
patientsCopy = patients.copy()
print('Empty values filled!\n')

# zbiór danych
x = patients.drop('Level', axis=1).to_numpy()
print('Set of important data:\n', x)

# zbiór etykiet
y = patients.loc[:, 'Level'].to_numpy()
print('Set of labels:\n', y)

# podział zbioru na dane treningowe i testowe
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


# uniwersalna metoda do trenowania i oceny modeli
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # trenuj model
    classifier.fit(feature_vector_train, label)

    # wygeneruj przewidywania modelu dla zbioru testowego
    predictions = classifier.predict(feature_vector_valid)

    # dokonaj ewaluacji modelu na podstawie danych testowych
    scores = list(metrics.precision_recall_fscore_support(predictions, y_test))
    score_vals = [scores[0][0], scores[1][0], scores[2][0], metrics.accuracy_score(predictions, y_test)]
    return score_vals


# MODEL 1 - regresja logistyczna
accuracy = train_model(linear_model.LogisticRegression(max_iter=250), x_train, y_train, x_test)
accuracy_compare = {'LR': accuracy}
print("LR: ", accuracy)

# MODEL 2 - Support Vector Machine
accuracy = train_model(svm.SVC(), x_train, y_train, x_test)
accuracy_compare['SVM'] = accuracy
print("SVM:", accuracy)

# MODEL 3 - Random Forest Tree
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=1, max_depth=1), x_train, y_train, x_test)
accuracy_compare['RF'] = accuracy
print("RF: ", accuracy)

# porównanie modeli
df_compare = pd.DataFrame(accuracy_compare, index=['precision', 'recall', 'f1 score', 'accuracy'])
df_compare.plot(kind='bar')
plt.show()

# działania korygujące - zastosowanie sieci neuronowej

# MODEL 4 - neural network

mlp = MLPClassifier(hidden_layer_sizes=(10, 5, 2), max_iter=250)
accuracy = train_model(mlp, x_train, y_train, x_test)
accuracy_compare['neural network'] = accuracy
print("neural network", accuracy)

# działania korygujące - hiperparametry

# MODEL 5 - Support Vector Machine
accuracy = train_model(svm.SVC(), x_train, y_train, x_test)
accuracy_compare['SVM'] = accuracy
print("SVM gamma='auto'", accuracy)

# MODEL 6 - Support Vector Machine
accuracy = train_model(svm.SVC(kernel='sigmoid'), x_train, y_train, x_test)
accuracy_compare['SVM'] = accuracy
print("SVM kernel='sigmoid'", accuracy)

# MODEL 7 - Support Vector Machine
accuracy = train_model(svm.SVC(degree=3), x_train, y_train, x_test)
accuracy_compare['SVM'] = accuracy
print("SVM degree=4", accuracy)
