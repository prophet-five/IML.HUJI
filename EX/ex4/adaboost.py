"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
import ex4_tools as t4
import matplotlib.pyplot as plt


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
            self.h[i] = weak_learner
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


def error(model, t, X, y):
    return model.error(X, y, t)


def plot_q13(T, tr_err, te_err, noise):
    plt.figure()
    plt.plot(T, tr_err, label='Training Error')
    plt.plot(T, te_err, label='Testing Error')
    plt.legend()
    plt.title(f"Q13: Adaboost with decision stumps\n"
              f"with noise: {noise}")
    plt.show()


def q13_generate_data(noise=0):
    train_X, train_y = t4.generate_data(5000, noise)
    test_X, test_y = t4.generate_data(200, noise)
    ada = AdaBoost(t4.DecisionStump, 500)
    ada.train(train_X, train_y)
    T = np.arange(1, 501)
    T = T[..., np.newaxis]
    tr_err = np.apply_along_axis(lambda t, X, y: error(ada, t[0], X, y), 1, T,
                                 train_X, train_y)
    te_err = np.apply_along_axis(lambda t, X, y: error(ada, t[0], X, y), 1, T,
                                 test_X, test_y)
    plot_q13(T, tr_err, te_err, noise)


def q14_generate_data(noise=0):
    train_X, train_y = t4.generate_data(5000, noise)
    test_X, test_y = t4.generate_data(200, noise)
    ada = AdaBoost(t4.DecisionStump, 500)
    ada.train(train_X, train_y)
    T = enumerate([5, 10, 50, 100, 200, 500])
    plt.suptitle(
        f"Q14: decisions of learned qualifiers with noise: {noise}, and increasing Ts")
    for i, t in T:
        plt.subplot(3, 2, i + 1)
        t4.decision_boundaries(ada, test_X, test_y, t)
    plt.show()


def q15_T_min_error(noise=0):
    train_X, train_y = t4.generate_data(5000, noise)
    test_X, test_y = t4.generate_data(200, noise)
    ada = AdaBoost(t4.DecisionStump, 500)
    ada.train(train_X, train_y)
    t = np.arange(1, 501)
    t = t[..., np.newaxis]
    err = np.apply_along_axis(lambda i: ada.error(test_X, test_y, i[0]), 1, t)
    minimizer = np.argmin(err)
    t4.decision_boundaries(ada, train_X, train_y, minimizer)
    err_min = err[minimizer]
    plt.suptitle(
        f"Q15: T that minimizes error: {minimizer}, Error: {err_min}, noise: {noise}")
    plt.show()


def q16(noise=0):
    train_X, train_y = t4.generate_data(5000, noise)
    test_X, test_y = t4.generate_data(200, noise)
    ada = AdaBoost(t4.DecisionStump, 500)
    weights = ada.train(train_X, train_y)
    plt.subplot(1, 2, 1)
    t4.decision_boundaries(ada, train_X, train_y, 500, weights)
    plt.subplot(1, 2, 2)
    weights = 10 * (weights / weights.max())
    t4.decision_boundaries(ada, train_X, train_y, 500, weights)
    plt.suptitle(
        f'Q16: Training a set of  size proportional to its weight with noise: {noise}\n'
        f'right - not normalized, left normalized')
    plt.show()


# for noise in [0, 0.01, 0.4]:
    # q13_generate_data(noise)
    # q14_generate_data(noise)
    # q15_T_min_error(noise)
    # q16(noise)
