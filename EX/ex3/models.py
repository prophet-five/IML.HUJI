from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self):
        _trained_model = None

    @abstractmethod
    def fit(self, X, y):
        """
        Learn the parameters of the model and store the trained model.
        :param X: The training set
        :param y: The response vector with respect to X.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        predict the label of each sample
        :param X: Unlabeled test set
        :return: prediction of the response
        """
        pass

    @abstractmethod
    def score(self, X, y):
        """
        Calculating score of the test set in different parameters.
        :param X: test set
        :param y: True labels of X
        :return: A dictionary with the fields: <number of samples in the test
        set, error rate, accuracy, FPR, TPR,precision,specificity.
        """


class Perceptron:
    pass
