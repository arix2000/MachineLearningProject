from sklearn import metrics, linear_model, svm, ensemble
from sklearn.neural_network import MLPClassifier


class ModelTrainer:
    accuracy_compare = {}
    max_iterations = 800
    svm_key = 'SVM'

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid):
        classifier.fit(feature_vector_train, label)

        predictions = classifier.predict(feature_vector_valid)

        scores = list(metrics.precision_recall_fscore_support(predictions, self.y_test))
        score_vals = [scores[0][0], scores[1][0], scores[2][0], metrics.accuracy_score(predictions, self.y_test)]
        return score_vals

    def get_trained_models_comparison(self):
        accuracy = self.train_model(linear_model.LogisticRegression(), self.x_train,
                                    self.y_train,
                                    self.x_test)
        accuracy_compare = {'LR': accuracy}
        print("LR: ", accuracy)

        accuracy = self.train_model(svm.SVC(), self.x_train, self.y_train, self.x_test)
        accuracy_compare['SVM'] = accuracy
        print("SVM:", accuracy)

        accuracy = self.train_model(ensemble.RandomForestClassifier(), self.x_train,
                                    self.y_train, self.x_test)
        accuracy_compare['RF'] = accuracy
        print("RF: ", accuracy)
        self.accuracy_compare = accuracy_compare
        return accuracy_compare

    def correct_models(self):
        mlp = MLPClassifier(hidden_layer_sizes=(10, 6, 4), max_iter=self.max_iterations)
        accuracy = self.train_model(mlp, self.x_train, self.y_train, self.x_test)
        self.accuracy_compare['neural network'] = accuracy
        print("neural network", accuracy)

        accuracy = self.train_model(svm.SVC(gamma='auto'), self.x_train, self.y_train, self.x_test)
        self.accuracy_compare['SVM'] = accuracy
        print("SVM gamma='auto'", accuracy)

        accuracy = self.train_model(svm.SVC(kernel='sigmoid'), self.x_train, self.y_train, self.x_test)
        self.accuracy_compare['SVM'] = accuracy
        print("SVM kernel='sigmoid'", accuracy)

        accuracy = self.train_model(svm.SVC(degree=4), self.x_train, self.y_train, self.x_test)
        self.accuracy_compare['SVM'] = accuracy
        print("SVM degree=4", accuracy)
        return self.accuracy_compare

    def correct_svm_model(self, classifier, print_text):
        accuracy = self.train_model(classifier, self.x_train, self.y_train, self.x_test)
        self.accuracy_compare[self.svm_key] = accuracy
        print(print_text, accuracy)
