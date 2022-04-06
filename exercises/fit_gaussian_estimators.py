from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    ran = np.random.normal(10, 1, 1000)
    x = UnivariateGaussian().fit(ran)
    print(f"( {x.mu_},{x.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    x1 = []
    y1 = []
    for i in range(0, 1000, 10):
        y1.append(np.abs(UnivariateGaussian().fit(ran[0:i + 10]).mu_ - 10))
        x1.append(i + 10)
    go.Figure([go.Scatter(x=x1, y=y1, mode='markers+lines',
                          marker=dict(color="black"),
                          showlegend=False)]).show()

    x2 = ran
    y2 = x.pdf(ran)

    go.Figure([go.Scatter(x=x2, y=y2, mode='markers',
                          marker=dict(color="black"),
                          showlegend=False)]).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    ran = np.random.multivariate_normal(mu, sigma, 1000)
    x = MultivariateGaussian().fit(ran)
    print(x.mu_)
    print(x.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    ma = np.ndarray([200, 200])
    Mue = mu
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_matrix = np.zeros((len(f1), len(f3)))
    for f1_index, f1_val in enumerate(f1):
        for f3_index, f3_val in enumerate(f3):
            Mue = np.array([f1_val, 0, f3_val, 0])
            log_matrix[f1_index][
                f3_index] = MultivariateGaussian.log_likelihood(Mue, np.array(
                sigma), ran)

    heatmap = go.Figure(
        data=go.Heatmap(x=f1, y=f3, z=log_matrix, hoverongaps=False),
        layout=go.Layout(title=r"$\text{Log Likelihood}$",
                         xaxis_title="$\\text{f3}$",
                         yaxis_title="r$\\text{f1}$", height=500))
    heatmap.show()

    # Question 6 - Maximum likelihood
    max_0, max_2 = np.unravel_index(np.argmax(log_matrix), log_matrix.shape)
    print(np.linspace(-10, 10, 200)[max_0], np.linspace(-10, 10, 200)[max_2])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
