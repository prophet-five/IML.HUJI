from abc import ABC, abstractmethod
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Classifier(ABC):
    """An abstract class defining a classification model"""

    def __init__(self):
        self._trained_model = None

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

    def score(self, X, y):
        """
        Calculating score of the test set in different parameters.
        :param X: test set
        :param y: True labels of X
        :return: A dictionary with the fields: <number of samples in the test
        set, error rate, accuracy, FPR, TPR,precision,specificity.
        """
        score_dict = {}
        sample_size = y.size
        score_dict["num_samples"] = sample_size
        pred = self.predict(X)
        # prediction assessment
        tp = np.sum(((pred == 1) & (y == 1)).astype(np.int8))
        tn = np.sum(((pred == -1) & (y == -1)).astype(np.int8))
        fp = np.sum(((pred == 1) & (y == -1)).astype(np.int8))
        fn = np.sum(((pred == -1) & (y == 1)).astype(np.int8))
        score_dict["error"] = (fp + fn) / sample_size
        score_dict["accuracy"] = 1 - score_dict["error"]
        score_dict["FPR"] = fp / (fp + tn)
        score_dict["TPR"] = tp / (tp + fn)
        score_dict["precision"] = tp / (tp + fp)
        score_dict["specificty"] = 1 - score_dict[
            "FPR"]  # The typo is defined in the ex description
        return score_dict

    def get_trained_model(self):
        return self._trained_model


class Perceptron(Classifier):
    def __init__(self):
        super().__init__()
        self._trained_model = None

    def fit(self, X, y):
        # add initial labels
        init_lab = np.ones((X.shape[0], 1))
        # Add initialized labels vector
        X_lab = np.concatenate((X, init_lab), axis=1)
        # model weights
        w = np.zeros(X_lab.shape[1])
        while np.any(np.sign(X_lab @ w) - y):
            # find mislabeled values
            mislabels = np.nonzero(np.sign(X_lab @ w) - y)[0][0]
            w = w + X_lab[mislabels] * y[mislabels]
        self._trained_model = w

    def predict(self, X):
        init_lab = np.ones((X.shape[0], 1))
        # Add initialized labels vector
        X_lab = np.concatenate((X, init_lab), axis=1)
        prediction = np.sign(X_lab @ self._trained_model)
        # TODO can be 0s?
        return prediction


class LDA(Classifier):
    def __init__(self):
        super().__init__()
        self._trained_model = None

    def fit(self, X, y):
        # calculate variables
        y_mean = np.array([X[y == 1].mean(axis=0), X[y == -1].mean(axis=0)]).T
        diff_pos = X[y == 1] - y_mean[:, 0]
        diff_neg = X[y == -1] - y_mean[:, 1]
        sig = (diff_pos.T @ diff_pos + diff_neg.T @ diff_neg) / y.size
        # Calculate the bias
        y_prob = np.log(np.array([(y == 1).mean(), (y == -1).mean()]))
        bias = (-0.5) * np.diag(
            y_mean.T @ np.linalg.inv(sig) @ y_mean) + y_prob
        self._trained_model = (y_mean, sig, bias)

    def predict(self, X):
        X = X.T
        y_mean, sig, bias = self._trained_model
        prediction = (-2) * np.argmax(X.T @ np.linalg.inv(sig) @ y_mean + bias,
                                      axis=1) + 1
        return prediction


class SVM(Classifier):
    def __init__(self):
        super().__init__()
        self._trained_model = None

    def fit(self, X, y):
        svm = SVC(C=1e10, kernel='linear')
        svm.fit(X, y)
        self._trained_model = svm

    def predict(self, X):
        return self._trained_model.predict(X)

    def score(self, X, y):
        """override Classifier's (super) score"""
        return self._trained_model.score(X, y)


class Logistic(Classifier):
    def __init__(self):
        super().__init__()
        self._trained_model = None

    def fit(self, X, y):
        logis_model = LogisticRegression(solver='liblinear')
        logis_model.fit(X, y)
        self._trained_model = logis_model

    def predict(self, X):
        return self._trained_model.predict(X)

    def score(self, X, y):
        """override Classifier's (super) score"""
        return self._trained_model.score(X, y)


class DecisionTree(Classifier):
    def __init__(self):
        super().__init__()
        self._trained_model = None

    def fit(self, X, y):
        # TODO: test max_depth parameter
        dt_model = DecisionTreeClassifier(max_depth=12)
        dt_model.fit(X, y)
        self._trained_model = dt_model

    def predict(self, X):
        return self._trained_model.predict(X)

    def score(self, X, y):
        """override Classifier's (super) score"""
        return self._trained_model.score(X, y)
