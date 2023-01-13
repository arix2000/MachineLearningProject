from sklearn import metrics, linear_model, svm, ensemble
from sklearn.neural_network import MLPClassifier


class ModelTrainer:
    accuracy_compare = None
    max_iterations = 250
    svm_key = 'SVM'

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    # uniwersalna metoda do trenowania i oceny modeli
    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid):
        # trenuj model
        classifier.fit(feature_vector_train, label)

        # wygeneruj przewidywania modelu dla zbioru testowego
        predictions = classifier.predict(feature_vector_valid)

        # dokonaj ewaluacji modelu na podstawie danych testowych
        scores = list(metrics.precision_recall_fscore_support(predictions, self.y_test))
        score_vals = [scores[0][0], scores[1][0], scores[2][0], metrics.accuracy_score(predictions, self.y_test)]
        return score_vals

    def get_trained_models_comparison(self):
        # MODEL 1 - regresja logistyczna
        accuracy = self.train_model(linear_model.LogisticRegression(max_iter=self.max_iterations), self.x_train,
                                    self.y_train,
                                    self.x_test)
        accuracy_compare = {'LR': accuracy}
        print("LR: ", accuracy)

        # MODEL 2 - Support Vector Machine
        accuracy = self.train_model(svm.SVC(), self.x_train, self.y_train, self.x_test)
        accuracy_compare['SVM'] = accuracy
        print("SVM:", accuracy)

        # MODEL 3 - Random Forest Tree
        accuracy = self.train_model(ensemble.RandomForestClassifier(n_estimators=1, max_depth=1), self.x_train,
                                    self.y_train, self.x_test)
        accuracy_compare['RF'] = accuracy
        print("RF: ", accuracy)
        self.accuracy_compare = accuracy_compare
        return accuracy_compare

    def correct_models(self):
        # działania korygujące - zastosowanie sieci neuronowej

        # MODEL 4 - neural network
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5, 2), max_iter=self.max_iterations)
        accuracy = self.train_model(mlp, self.x_train, self.y_train, self.x_test)
        self.accuracy_compare['neural network'] = accuracy
        print("neural network", accuracy)

        # działania korygujące - hiperparametry

        # MODEL 5 - Support Vector Machine
        self.correct_svm_model(svm.SVC(), "SVM gamma='auto'")

        # MODEL 6 - Support Vector Machine
        self.correct_svm_model(svm.SVC(kernel='sigmoid'), "SVM kernel='sigmoid'")

        # MODEL 7 - Support Vector Machine
        self.correct_svm_model(svm.SVC(degree=3), "SVM degree=3")

    def correct_svm_model(self, classifier, print_text):
        accuracy = self.train_model(classifier, self.x_train, self.y_train, self.x_test)
        self.accuracy_compare[self.svm_key] = accuracy
        print(print_text, accuracy)
