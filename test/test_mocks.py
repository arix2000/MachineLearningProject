import pandas as pd
from pandas import DataFrame

smoking = [1, 2, 4, 5, 2, 6, 2, 8, 6]
numbers = [10, 12, 14, 71, None, None, 23, 52, None]
gender = [1, 0, 0, 1, 1, 0, 1, 0, 1]
age = [54, 23, 42, 35, 18, 56, 86, 25, 30]
data = {'Gender': gender, 'Numbers': numbers, 'Smoking': smoking, 'Age': age}
patients_test_2 = DataFrame(data)

smoking = [1, 2, 4, 5, 2, 6, 2, 8, 6]
numbers = [10, 12, 14, 71, 8, 2, 23, 52, 210]
gender = [1, 0, 0, 1, 1, 0, 1, 0, 1]
age = [54, 23, 42, 35, 18, 56, 86, 25, 30]
data = {'Gender': gender, 'Numbers': numbers, 'Smoking': smoking, 'Age': age}
patients_test_1 = DataFrame(data)


def get_patients_test() -> DataFrame:
    patients: DataFrame = pd.read_csv('cancer_patient_data_sets.csv')
    patients = patients.replace(to_replace="Low", value=0)
    patients = patients.replace(to_replace="Medium", value=1)
    patients = patients.replace(to_replace="High", value=2)
    patients.drop(
        ['index', 'Patient Id', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet',
         'Obesity',
         'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing',
         'Swallowing Difficulty',
         'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough'], axis=1, inplace=True)
    return patients
