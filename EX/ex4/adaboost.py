"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
import ex4_tools as t4


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        samples_num = X.shape[0]
        sample_weights = np.ones(samples_num) / samples_num

        # run over all base learners
        for i in range(self.T):
            weak_learner = self.WL(sample_weights, X, y)
            # predict
            y_pred = weak_learner.predict(X)
            diff = (y != y_pred)
            tot_err = sample_weights[
                diff].sum()  # incorrect predictions' weights
            # adjust weights
            self.w[i] = 0.5 * np.log((1 / float(tot_err)))
            sign = (y * y_pred) * (-1)
            # Update weights
            sample_weights = sample_weights * np.exp(sign * self.w[i])
            sample_weights = sample_weights / np.sum(
                sample_weights)  # normalize
        return sample_weights

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        y_pred = np.zeros(X.shape[0])
        # y_hat = (self.h[t].predict(X) * self.w[t])
        for i in range(max_t):
            y_pred += self.w[i] * self.h[i].predict(X)
        return np.sign(y_pred)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_pred = self.predict(X, max_t)
        err = np.sum(y_pred != y) / y.shape[0]
        return err


# def generate_data(sample_num=5000, noise_ratio=0, T=500):
#     train = k


def q13_generate_data():
    train_X, train_y = t4.generate_data(5000, 0)
    test_X, test_y = t4.generate_data(200, 0)
    ada = AdaBoost(t4.DecisionStump, 500)

    pass
