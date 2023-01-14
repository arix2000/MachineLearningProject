import unittest

from test.test_mocks import patients_test_2, patients_test_1
from utils.analitycs import Analytics


class AnalyticsTest(unittest.TestCase):
    analytics = Analytics(patients_test_1)
    analytics2 = Analytics(patients_test_2)

    def test_is_there_any_nulls(self):
        result = self.analytics.is_there_any_nulls()
        self.assertFalse(result)
        result = self.analytics2.is_there_any_nulls()
        self.assertTrue(result)

    def test_show_gender_distribution(self):
        dist = self.analytics.gender_distribution()
        axis = dist.gca().patches
        heights = [float(x.get_height()) for x in axis]
        self.assertListEqual(heights, [4.0, 5.0])

    def test_smokers_histogram(self):
        dist = self.analytics.smokers_histogram()
        axis = dist.gca().patches
        heights = [float(x.get_height()) for x in axis]
        self.assertListEqual(heights, [1.0, 3.0, 0.0, 1.0, 1.0, 2.0, 1.0])

    def test_model_comparison(self):
        accuracy_compare = {'LR': [0.9047619047619048, 0.9661016949152542, 0.9344262295081968, 0.86],
                            'RF': [0.6102930981230, 0.509234910923409, 0.81910912338291, 0.755],
                            'SVM': [0.9047619047619048, 0.8028169014084507, 0.8507462686567164, 0.805]}
        comparison = self.analytics.model_comparison(accuracy_compare)
        axis = comparison.gca().patches
        heights = [float(x.get_height()) for x in axis]
        self.assertListEqual(heights, [0.9047619047619048, 0.9661016949152542, 0.9344262295081968, 0.86, 0.610293098123,
                                       0.509234910923409, 0.81910912338291, 0.755, 0.9047619047619048,
                                       0.8028169014084507, 0.8507462686567164, 0.805])
