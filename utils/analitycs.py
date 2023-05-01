import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


class Analytics:
    def __init__(self, patients):
        self.patients = patients

    def print_overall_data_info(self):
        print(self.patients.describe())
        print(self.patients.info())

    def is_there_any_nulls(self):
        print("\nWhether there are null values in the data set?")
        return self.patients.isnull().values.any()

    def gender_distribution(self):
        gender_dist = self.patients['Gender'].hist(bins=2, label=["Female", "Male"])
        plt.xlabel("gender")
        plt.ylabel("number of people")
        gender_dist.set_title('GENDER DISTRIBUTION')
        plt.gca().xaxis.set_major_locator(ticker.FixedLocator([1, 2]))
        plt.gca().set_xticklabels(['0', '1'])
        plt.xticks(np.arange(min(self.patients['Gender']), max(self.patients['Gender']) + 1, 1))
        return plt

    def smokers_histogram(self):
        smokers_hist = self.patients['Smoking'].hist(bins=7)
        title = 'SMOKERS IN THE RESEARCH GROUP'
        plt.xlabel("level of smoking")
        plt.ylabel("number of people")
        smokers_hist.set_title(title)
        return plt

    def age_distribution(self):
        age_dist = self.patients['Age'].plot.box()
        title = 'DATA DISTRIBUTION (BASED ON AGE)'
        age_dist.set_title(title)
        return plt

    def correlation_matrix(self):
        correlations = self.patients.corr()
        plt.subplots(figsize=(24, 24))
        colormap = sns.color_palette("BrBG", 10)
        sns.heatmap(correlations,
                    cmap=colormap,
                    annot=True,
                    fmt=".2f")
        return plt

    def model_comparison(self, accuracy_compare):
        df_compare = pd.DataFrame(accuracy_compare, index=['precision', 'recall', 'f1 score', 'accuracy'])
        df_compare.plot(kind='bar')
        return plt
