from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from IMLearn.utils import utils
from IMLearn.learners.regressors import linear_regression


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # house_prise = pd.read_csv(filename)
    # house_prise.drop("id", axis=1)
    # house_prise.drop("date", axis=1)
    # house_prise.drop("lat", axis=1)
    # house_prise.drop("long", axis=1)
    # house_prise.drop("zipcode", axis=1)
    # house_prise["zipcode"] = house_prise["zipcode"].astype(int)
    # house_prise = house_prise.loc[~((house_prise['price'] > 0) | (house_prise['bedrooms'] == 0))]
    #price >0
    #
    # raise NotImplementedError()
    # data_f = pd.read_csv(filename).dropna().drop_duplicates()
    # for feature in ["id", "lat", "date", "long"]:
    #     data_f = data_f.drop(labels=feature, axis=1)
    #
    # data_f["zipcode"] = data_f["zipcode"].astype(int)  # some of them as string
    #
    # data_f = data_f.loc[
    #     (data_f[['price', "floors", 'condition', 'grade', 'sqft_above', 'yr_built',
    #          'sqft_living15', 'sqft_lot15']] > 0).all(axis=1)]
    # data_f = data_f.loc[(data_f[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    #                  'view', 'waterfront', 'sqft_basement',
    #                  'yr_renovated']] >= 0).all(
    #     axis=1)]
    # data_f = data_f.loc[data_f['grade'] < 14]
    # data_f = data_f.loc[data_f['condition'] <= 5]
    # data_f = data_f.loc[data_f['waterfront'] <= 1]
    #
    # return data_f.drop('Temp', axis=1), data_f['Temp']
    data_f = pd.read_csv(filename).dropna().drop_duplicates()
    for feature in ["id", "date", "lat", "long"]:
        data_f = data_f.drop(labels=feature, axis=1)

    data_f["zipcode"] = data_f["zipcode"].astype(int)  # some of them as string

    data_f = data_f.loc[
        (data_f[['price', "floors", 'condition', 'grade', 'sqft_above', 'yr_built',
             'sqft_living15', 'sqft_lot15']] > 0).all(axis=1)]
    data_f = data_f.loc[(data_f[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                     'view', 'waterfront', 'sqft_basement',
                     'yr_renovated']] >= 0).all(
        axis=1)]
    data_f = data_f.loc[data_f['grade'] < 14]
    data_f = data_f.loc[data_f['condition'] <= 5]
    data_f = data_f.loc[data_f['waterfront'] <= 1]

    return data_f.drop('price', axis=1), data_f['price']


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # p = (np.cov(X, y))/(np.std(X)*np.std(y))
    x_1 = X["sqft_living"].values
    x_2 = X["zipcode"].values

    p1 = ((np.cov(x_1, y)) / (np.std(x_1) * np.std(y)))[0][1]
    p2 = ((np.cov(x_2, y)) / (np.std(x_2) * np.std(y)))[0][1]
    fg = make_subplots(rows=1, cols=2, start_cell="bottom-left")
    fg.add_traces([go.Scatter(x=x_1, y=y, mode="markers"),
                    go.Scatter(x=x_2, y=y, mode="markers")], rows=[1, 1],
                   cols=[1, 2])
    fg.update_xaxes(title_text=f"Rsqft_living, p={p1}", row=1, col=1)
    fg.update_xaxes(title_text=f"zipcode, p={p2}", row=1, col=2)
    fg.update_yaxes(title_text="Prices")
    fg.write_image(output_path)



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('C:\\Users\\harel\\imlP\\IML.HUJI\\datasets\\house_prices.csv')
    # raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "3.1.2 PearsonCorrelation.png")

    # Question 3 - Split samples into training- and testing sets.
    xTrain, yTrain, xTest, yTest = utils.split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    x = np.linspace(10, 100, 1)
    losses, avgLoss = [], []
    lR = linear_regression.LinearRegression()
    mean_pred, std_pred = [], []
    for i in range(10, 100):
        for _ in range(10):
            samplesP = xTrain.sample(frac=i / 100)
            lR.fit(samplesP.to_numpy(), y[samplesP.index].to_numpy())
            predict = lR.predict(xTest.to_numpy())
            loss = lR.loss(xTest.to_numpy(), yTest.to_numpy())
            losses.append(loss)

        avgLoss.append(np.average(losses))
        mean_pred.append(np.mean(losses))
        std_pred.append(np.std(losses))
        losses = []

    lR.fit(xTrain.to_numpy(), yTrain.to_numpy())
    predict = lR.predict(xTrain.to_numpy())
    loss = lR.loss(xTest.to_numpy(), yTest.to_numpy())
    avgLoss.append(loss)
    mean_pred = np.array(mean_pred)
    std_pred = np.array(std_pred)
    fg = go.Figure(
        data=[
            go.Scatter(x=list(range(1, len(avgLoss))), y=avgLoss,
                       mode="markers+lines",
                       name="average loss"),
            go.Scatter(x=list(range(1, len(avgLoss))),
                       y=mean_pred - 2 * std_pred, fill=None, mode="lines",
                       line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=list(range(1, len(avgLoss))),
                       y=mean_pred + 2 * std_pred, fill='tonexty',
                       mode="lines", line=dict(color="lightgrey"),
                       showlegend=False)],
        layout=go.Layout(title_text="MSE loss to size of training data",
                         xaxis={"title": "size of training data set"},
                         yaxis={"title": "MSE loss"}))

    #
    fg.write_image("3.1.4 mse.png")
