import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as dtimer

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def play_with_ds():
    """Q12"""
    idx = np.hstack((
        (np.where(y_train == 0)[0][:3], np.where(y_train == 1)[0][:3])))
    fig = plt.figure()
    imgs = x_train[idx, :, :]

    for i in range(len(idx)):
        fig.add_subplot(2, 3, i + 1)
        plt.imshow(imgs[i])
    plt.show()


def rearrange_data(X):
    """Q13"""
    dim1, dim2, dim3 = X.shape
    return X.reshape((dim1, dim2 * dim3))


def draw_points(m):
    # draw random indices
    idx = np.random.randint(y_train.size, size=m)
    y = y_train[idx]

    while np.all(y == 1) or np.all(y == 0):
        idx = np.random.randint(y.size, size=m)
        y = y_train[idx]
    return x_train[idx, :, :], y_train[idx]


def Q14_combined():
    ms = [50, 100, 300, 500]
    reps = 50
    all_models = {"Logistic": models.Logistic(), "SVM": models.SVM(),
                  "Tree": models.DecisionTree(),
                  "KNearest": KNeighborsClassifier(n_neighbors=5)}
    model_size = len(all_models)
    timers = np.zeros((model_size, model_size))
    mean_score = np.zeros((model_size, model_size))

    for i in range(len(ms)):
        for j in range(reps):
            X_rand, y_rand = draw_points(ms[i])
            # rearrange
            X_rand = rearrange_data(X_rand)
            x_test_new = rearrange_data(x_test)

            # fit
            idx = 0
            for model in all_models:
                # fit model
                stimer = dtimer()
                all_models[model].fit(X_rand, y_rand)
                score = all_models[model].score(x_test_new, y_test)
                time = dtimer() - stimer
                mean_score[idx, i] += score
                timers[idx, i] += time
                idx += 1

        mean_score[:, i] /= reps
        timers[:, i] /= reps
    for i, model in enumerate(all_models):
        print(f'{model} time:{timers[i]}')
        plt.plot(ms, mean_score[i], label=f"{model} mean accuracy")
    plt.xlabel("m")
    plt.ylabel("Accuracy")
    plt.title("Mean Accuracy as Function of m ")
    plt.legend()
    plt.show()
