import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def show_model_comparison(accuracy_compare):
    df_compare = pd.DataFrame(accuracy_compare, index=['precision', 'recall', 'f1 score', 'accuracy'])
    df_compare.plot(kind='bar')
    plt.show()


class Analytics:
    def __init__(self, patients):
        self.patients = patients

    def print_overall_data_info(self):
        print(self.patients.describe())
        print(self.patients.info())

    def is_there_any_nulls(self):
        print("\nWhether there are null values in the data set?")
        print(self.patients.isnull().values.any())

    def show_gender_distribution(self):
        gender_dist = self.patients['Gender'].hist(bins=2, label=["Female", "Male"])
        plt.xlabel("gender")
        plt.ylabel("number of people")
        title = 'GENDER DISTRIBUTION'
        gender_dist.set_title(title)
        plt.gca().set_xticklabels(['Male', 'Female'])
        plt.xticks(np.arange(min(self.patients['Gender']), max(self.patients['Gender']) + 1, 1))
        plt.show()

    def show_smokers_histogram(self):
        cancer_risk = self.patients['Smoking'].hist(bins=7)
        title = 'SMOKERS IN THE RESEARCH GROUP'
        plt.xlabel("level of smoking")
        plt.ylabel("number of people")
        cancer_risk.set_title(title)
        plt.show()

    def show_age_distribution(self):
        age_dist = self.patients['Age'].plot.box()
        title = 'DATA DISTRIBUTION (BASED ON AGE)'
        age_dist.set_title(title)
        plt.show()

    def show_correlation_matrix(self):
        correlations = self.patients.corr()
        plt.subplots(figsize=(24, 24))
        colormap = sns.color_palette("BrBG", 10)
        sns.heatmap(correlations,
                    cmap=colormap,
                    annot=True,
                    fmt=".2f")
        plt.show()

