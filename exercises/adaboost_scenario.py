import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaBoost.fit(train_X, train_y)
    train_er = [adaBoost.partial_loss(train_X, train_y, i)
                for i in range(n_learners)]
    test_er = [adaBoost.partial_loss(test_X, test_y, i)
               for i in range(n_learners)]
    fg = go.Figure([go.Scatter(x=np.arange(1, n_learners), y=train_er,
                               mode='lines', name='train error'),
                    go.Scatter(x=np.arange(1, n_learners), y=test_er,
                               mode='lines', name='test '
                                                  'error')]).update_layout(
        title="error of train and test as number of learners").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    # sub = make_subplots(2, 2, subplot_titles=[f"{t} learners" for t in T])
    # fg2 = go.scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
    #                  showlegend=False,
    #                  marker=dict(color=test_y))
    ps = make_subplots(2, 2, subplot_titles=[f"{t} learners" for t in T])

    fg22 = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                      showlegend=False,
                      marker=dict(color=test_y))
    col = [1, 2, 1, 2]
    row = [1, 1, 2, 2]
    for j, t in enumerate(T):
        ps.add_traces([decision_surface(lambda x:
                                        adaBoost.partial_predict(x, t),
                                        lims[0], lims[1],
                                        showscale=False), fg22], rows=row[j],
                      cols=col[j])
    ps.update_layout(title="decision surfaces").show()

    # Question 3: Decision surface of best performing ensemble
    err_min = np.argmin(test_er) + 1
    accr = 1 - test_er[err_min - 1]
    go.Figure(
        [decision_surface(lambda x: adaBoost.partial_predict(x, err_min),
                          lims[0], lims[1], showscale=False),
         fg22]).update_layout(title=f"decision surface of best"
                                    f" performing ensemble is {err_min}"
                                    f" with accuracy of {accr} ").show()

    # Question 4: Decision surface with weighted samples
    D = adaBoost.D_ / np.max(adaBoost.D_) * 5
    go.Figure([decision_surface(adaBoost.predict,
                                lims[0], lims[1],
                                showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                          showlegend=False,
                          marker=dict(color=test_y, size=D))]).update_layout(
        title="Decision surface with weighted samples").show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
    # raise NotImplementedError()
