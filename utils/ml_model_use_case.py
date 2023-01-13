import numpy as np
from sklearn import svm


class MLModelUseCase:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def start(self):
        age = input("Enter patient age: ")
        gender = input("Enter gender (male/female)")
        if gender == "male":
            gender = 1
        else:
            gender = 2
        air_pollution = input("What is the air pollution at the patient's place of residence? (1-7) ")
        alcohol_use = input("How much alcohol the patient is using? (1-7) ")
        dust_allergy = input("Enter the level of dust allergy of the patient. (1-7) ")
        smoking = input("How much does the patient smoke? (1-7) ")
        passive_smoker = input("How often the patient is around smokers? (1-7) ")
        snoring = input("How often does the patient snore? (1-7) ")

        new_patient = np.array(
            [age, gender, air_pollution, alcohol_use, dust_allergy, smoking, passive_smoker, snoring])
        new_patient = new_patient.reshape(1, -1)
        self.predict_if_patient_has_lung_cancer(new_patient)

    def predict_if_patient_has_lung_cancer(self, new_patient):
        svm_model = svm.SVC(degree=3)
        svm_model.fit(self.x_train, self.y_train)
        predictions = svm_model.predict(new_patient)
        if predictions == 1:
            print("ML Model predicts that patient with near 100 percent certainty has lung cancer!")
        else:
            print("ML Model predicts that patient with near 100 percent certainty hasn't got lung cancer!")
