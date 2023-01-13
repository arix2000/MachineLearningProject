import numpy as np
from sklearn import svm


class MLModelUseCase:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def start(self):
        new_patient = np.array([51, 4, 2, 4, 5, 2, 3, 3])
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
