from sklearn.model_selection import train_test_split

from utils.analitycs import Analytics
from utils.data_manager import get_patients, drop_useless_from, fill_empty_values
from utils.ml_model_use_case import MLModelUseCase
from utils.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings("ignore")

patients = get_patients()
print(patients.head())
print("Data imported correctly.")

analytics = Analytics(patients)

print(analytics.is_there_any_nulls())

analytics.gender_distribution().show()

analytics.smokers_histogram().show()

analytics.age_distribution().show()

drop_useless_from(patients)

analytics.print_overall_data_info()

analytics.correlation_matrix().show()

fill_empty_values(patients)

x = patients.drop('Level', axis=1).to_numpy()
print('Set of important data:\n', x)

y = patients.loc[:, 'Level'].to_numpy()
print('Set of labels:\n', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

model_trainer = ModelTrainer(x_train, x_test, y_train, y_test)
accuracy_compare = model_trainer.get_trained_models_comparison()

analytics.model_comparison(accuracy_compare).show()

model_trainer.correct_models()

MLModelUseCase(x_train, y_train).start()
