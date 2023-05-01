import unittest

import pandas as pd

from test.test_mocks import patients_test_2
from utils.data_manager import drop_useless_from, fill_empty_values


class DataManagerTest(unittest.TestCase):
    patients = pd.read_csv('cancer_patient_data_sets.csv')
    data_with_empty_values = patients_test_2

    def test_drop_useless_from(self):
        drop_useless_from(self.patients)
        print(self.patients.columns.to_list())
        self.assertListEqual(self.patients.columns.to_list(),
                             ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Genetic Risk',
                              'chronic Lung Disease', 'Smoking', 'Passive Smoker', 'Level'])

    def test_fill_empty_values(self):
        fill_empty_values(self.data_with_empty_values)
        self.assertFalse(self.data_with_empty_values.isnull().values.any())
