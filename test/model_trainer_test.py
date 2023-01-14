import unittest

from sklearn.model_selection import train_test_split

from test.test_mocks import get_patients_test
from utils.model_trainer import ModelTrainer

patients = get_patients_test()
x = patients.drop('Level', axis=1).to_numpy()
y = patients.loc[:, 'Level'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


class ModelTrainerTest(unittest.TestCase):
    trainer = ModelTrainer(x_train, x_test, y_train, y_test)

    def test_get_trained_models_comparison(self):
        dictionary = self.trainer.get_trained_models_comparison()
        for key in dictionary.keys():
            self.assertTrue(key == 'LR' or key == "RF" or key == "SVM")
        for valueList in dictionary.values():
            for value in valueList:
                self.assertTrue(value <= 1.0)

    def test_correct_models(self):
        dictionary = self.trainer.correct_models()
        print(dictionary)
        for key in dictionary.keys():
            self.assertTrue(key == 'neural network' or key == "SVM")
        for valueList in dictionary.values():
            for value in valueList:
                self.assertTrue(value <= 1.0)

