import numpy as np
from numpy.random import multivariate_normal
from matplotlib import pyplot as plt
import models


def draw_points(m):
    """
    :param m:
    :return: a pair X, y where X is m X 2 matrix where each column represents
    an i.i.d sample from the given distribution and y is its corresponding label
    """

    cov = np.eye(2)
    mean = np.array([0, 0])
    done = False
    while not done:
        X = multivariate_normal(mean, cov, m)
        y = np.sign(X @ np.array([[0.3, -0.5]]).T + 0.1)
        # check y in correct range
        if not np.all((y == 1) | (y == -1)):
            continue
        done = True
    return X, y


def apply_draw():
    """Q9"""
    for m in [5, 10, 15, 25, 70]:
        X, y = draw_points(m)
        y = np.squeeze(y)
        w = [0.3, -0.5]

        # True hypothesis hyperplane image
        th_hp = [-3 * w[0] / -w[1] + 0.1 / -w[1],
                 3 * w[0] / -w[1] + 0.1 / -w[1]]

        # set color map
        colors = np.empty(y.shape, dtype='U30')
        colors[y == 1] = "blue"
        colors[y != 1] = "orange"

        # init models
        perc_model = models.Perceptron()
        svm_model = models.SVM()
        perc_model.fit(X, y)
        svm_model.fit(X, y)

        # plot
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.plot([-3, 3], th_hp, label='True Hypothesis',
                 color='green')

        w = perc_model.get_trained_model()
        # True hypothesis hyperplane image
        th_hp = [-3 * w[0] / -w[1] + 0.1 / -w[1],
                 3 * w[0] / -w[1] + 0.1 / -w[1]]

        plt.plot([-3, 3], th_hp, label='Perceptron Hypothesis', color='purple')

        model = svm_model.get_trained_model()
        th_hp = [
            -3 * model.coef_[0, 0] / model.coef_[0, 1] + model.intercept_ /
            model.coef_[0, 1],
            3 * model.coef_[0, 0] / -model.coef_[0, 1] + model.intercept_ / -
            model.coef_[0, 1]]

        plt.plot([-3, 3], th_hp, label='SVM Hypothesis', color='sienna')
        plt.title(f'Q9: apply draw : m={m}')
        plt.legend()
        plt.show()


def test_q10():
    rep, k = 500, 10000  # repetitions, # of test points
    # init models
    all_models = {"Perceptron": models.Perceptron(), "SVM": models.SVM(),
                  "LDA": models.LDA()}
    mean_acc = {model: [] for model in all_models.keys()}
    batch_sizes = [5, 10, 15, 25, 70]
    for m in batch_sizes:
        # init accuracy dict
        acc = {model: 0 for model in all_models.keys()}
        for i in range(rep):
            print(i)
            X, y = draw_points(m)
            y = np.squeeze(y)
            while not (np.any(y == 1) and np.any(y == -1)):
                X, y = draw_points(m)
                y = np.squeeze(y)
            test_X, test_y = draw_points(k)
            test_y = np.squeeze(test_y)
            # fit models
            for model in all_models:
                all_models[model].fit(X, y)
                acc[model] += np.sum(
                    test_y == all_models[model].predict(test_X)) / k
        for model in mean_acc:
            mean_acc[model].append(acc[model] / rep)
    for model in mean_acc:
        plt.plot(batch_sizes, mean_acc[model], label=f"mean accuracy: {model}")
        plt.xlabel("m")
        plt.ylabel("Accuracy")
        plt.title("Mean accuracy as function of m")
        plt.legend()
    plt.show()
