import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd


def fit_linear_regression(X, y):
    """Q9"""
    X_dag = np.linalg.pinv(X)
    w = X_dag @ y
    S = np.linalg.svd(X, compute_uv=False)
    return w, S


def predict(X, w):
    """Q10"""
    return X @ w


def mse(y, y_hat):
    """Q11"""
    return np.mean((y_hat - y) ** 2)


def load_data(path):
    """Q12/13"""
    # Read csv
    data = pd.read_csv(path)
    # remove negatives from positive-only fields
    positives = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
                 'sqft_lot', 'floors', 'price', 'waterfront', 'view',
                 'condition', 'grade', 'sqft_above', 'sqft_basement',
                 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15',
                 'sqft_lot15']
    crits = pandas.Series(np.ones(data.shape[0], dtype=bool))
    for crit in positives:
        crits = crits & (data[crit] >= 0)
    # Filter by valid ranges
    valid_ranges = {'view': (5,), 'condition': (1, 6), 'grade': (1, 14),
                    'waterfront': (0, 2)}
    for crit, ran in valid_ranges.items():
        crits = crits & (data[crit].isin(range(*ran)))
    data = data[crits]
    # Remove duplicate values
    data = data.drop_duplicates()
    # Hot encoding
    data = pd.get_dummies(data, columns=['zipcode'])
    # separate
    prices = data['price']
    data = data.drop(columns=['id', 'price', 'lat', 'long', 'date'])

    data.insert(0, 'intercept', 1, True)
    return data, prices


def plot_singular_values(vals):
    """Q14"""
    sorted(vals, reverse=True)
    plt.figure()
    plt.title('14) scree-plot for singular values')
    plt.xlabel('Index')
    plt.ylabel('Singular Values')
    plt.plot(np.arange(1, len(vals) + 1), vals, '-x')
    plt.show()


def stitch_together(path):
    """Q15"""
    # Load data
    X, prices = load_data(path)
    # Remove dummies
    X = X.loc[:, ~(X.columns.str.contains('zipcode_', case=False))].drop(
        "intercept", 1)

    S = np.linalg.svd(X, compute_uv=False)
    plot_singular_values(S)


def assign_model_data(X, y, ratio=0.25):
    """Q16 p.1"""
    X, y = X.sample(frac=1), y.reindex_like(X)
    n = round((1 - ratio) * len(y))
    train = (X[0:n], y[0:n])
    test = (X[n:], y[n:])
    return train, test


def fit_model(path):
    """Q16 p.2"""
    # Load data
    X, y = load_data(path)
    # Divide to train/test sets
    train, test = assign_model_data(X, y)
    mse_lst = []
    frac = len(train[0]) / 100
    for p in range(1, 101):
        rows = round(p * frac)
        w, _ = fit_linear_regression(train[0][:rows], train[1][:rows])
        pred = predict(test[0], w)
        mse_lst.append(mse(test[1], pred))
    plt.figure()
    plt.title("16) MSE with respect to % of the test set size")
    plt.xlabel("%")
    plt.ylabel("MSE")
    plt.plot(np.arange(1, 101), np.array(mse_lst), '-x')
    plt.show()


def feature_evaluation(X, y):
    """Q17"""
    # take features
    feats = X.columns[1:16]
    # evaluate
    for feat in feats:
        cov_mat = np.cov(X[feat], y)
        # Pearson correlation
        denom = np.std(X[feat]) * np.std(y)
        p_cor = cov_mat[0][1] / denom
        plt.scatter(X[feat], y)
        plt.title(
            f".\nPrice affected by feature: {feat} \n "
            f"Pearson correlation={p_cor}\n")
        plt.xlabel(f"{feat}")
        plt.ylabel("Price")
        plt.show()
