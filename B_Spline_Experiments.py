
from scipy.interpolate import LSQUnivariateSpline, BSpline, splrep
from csaps import csaps
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mean_square_error(
    predicted : np.ndarray,
    actual : np.ndarray
) -> float:
    '''
    mean_square_error:
        Return mean square error of residuals
        Both input array must be of the same size.

        Inputs:
            predicted : np.ndarray, required
                Predicted values of regression model
            actual : np.ndarray, required
                Actual values of regression model

        Outputs:
            mean_square_error : float
                Mean square error of model
    '''
    residuals = np.subtract(predicted, actual)
    square_error = np.square(residuals)
    mean_square_error = np.mean(square_error)
    return mean_square_error

def leave_one_out_cross_validation_c_spline(
    x : np.ndarray,
    y : np.ndarray,
    regression_model : callable,
    kwargs : dict
) -> float:
    '''
    leave_one_out_cross_validation_c_spline():
        Perform leave one out cross validation on a selected model.

        Inputs:
            x : np.ndarray, required
                Independent variable used to train the model
            y : np.ndarray, required
                Dependent variable used to train the model
            regression_model : callable, required
                Object to fit data into a regression model
            kwargs : dict, required
                Special parameters passed to the callable object
        Output:
            mean_square_error : float
                Mean square error of model
    '''
    predicted = []
    for i in range(0, len(x)):
        x_train = np.delete(x.copy(), i)
        y_train = np.delete(y.copy(), i)
        model = regression_model(x_train, y_train, **kwargs)
        prediction = model.__call__(x[i])
        predicted.append(prediction)
    predicted = np.array(predicted)

    mse = mean_square_error(predicted, y)
    return mse

def leave_one_out_cross_validation_b_spline(
    x : np.ndarray,
    y : np.ndarray,
    regression_model_1 : callable,
    regression_model_2 : callable,
    kwargs : dict
) -> float:
    '''
    leave_one_out_cross_validation_b_spline():
        Perform leave one out cross validation on a selected model.

        Inputs:
            x : np.ndarray, required
                Independent variable used to train the model
            y : np.ndarray, required
                Dependent variable used to train the model
            regression_model_1 : callable, required
                Object to fit data into a regression model
            regression_model_2 : callable, required
                Object to fit data into a regression model
            kwargs : dict, required
                Special parameters passed to the callable object
        Output:
            mean_square_error : float
                Mean square error of model
    '''
    predicted = []
    for i in range(0, len(x)):
        x_train = np.delete(x.copy(), i)
        y_train = np.delete(y.copy(), i)
        model_1 = regression_model_1(x_train, y_train, **kwargs)
        model_2 = regression_model_2(*model_1)
        prediction = model_2.__call__(x[i])
        predicted.append(prediction)
    predicted = np.array(predicted)

    mse = mean_square_error(predicted, y)
    return mse

def leave_one_out_cross_validation_gaussian(
    x : np.ndarray,
    y : np.ndarray,
    regression_model : callable,
    kwargs : dict
) -> float:
    '''
    leave_one_out_cross_validation_gaussian():
        Perform leave one out cross validation on a selected model.

        Inputs:
            x : np.ndarray, required
                Independent variable used to train the model
            y : np.ndarray, required
                Dependent variable used to train the model
            regression_model : callable, required
                Object to fit data into a regression model
            kwargs : dict, required
                Special parameters passed to the callable object
        Output:
            mean_square_error : float
                Mean square error of model
    '''
    predicted = []
    for i in range(0, len(x)):
        x_train = np.delete(x.copy(), i).reshape(-1, 1)
        y_train = np.delete(y.copy(), i).reshape(-1, 1)
        model = regression_model(**kwargs).fit(x_train, y_train)
        prediction = model.predict([[x[i]]])
        predicted.append(prediction[0][0])
    predicted = np.array(predicted)

    mse = mean_square_error(predicted, y)
    return mse

if __name__ == "__main__":
    filename = 'data/P04.csv'
    data = pd.read_csv(fr'{filename}')

    years_df = data.copy()
    years_df.columns = ['year', 'energy']
    x = years_df.iloc[:, 0].values
    y = years_df.iloc[:, 1].values

    ## ---------------------------------------------------------
    ## Cubic Spline
    ## ---------------------------------------------------------
    list_of_mse = []
    for num_of_knots in range(6, 16):
        knots = np.linspace(np.min(x), np.max(x), num_of_knots)[1: -1]
        kwargs = {
            't': knots,
            'k' : 3
        }
        mse = leave_one_out_cross_validation_c_spline(x, y, LSQUnivariateSpline, kwargs)
        list_of_mse.append(mse)

    knot_counts = np.expand_dims(np.arange(6, 16), axis = 1)
    mse_array = np.expand_dims(np.array(list_of_mse), axis = 1)
    mse_per_knot = np.hstack((knot_counts, mse_array))
    mse_per_knot_df = pd.DataFrame(mse_per_knot, columns = ['knots', 'mse'])

    plt.title('Cubic Spline (LOOCV)')
    sns.lineplot(data = mse_per_knot_df, x = 'knots', y = 'mse')
    plt.savefig(fname = 'figure_1.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    best_knot_count = np.argmin(mse_array) + np.min(knot_counts)
    knots = np.linspace(np.min(x), np.max(x), best_knot_count)[1: -1]
    cubic_spline = LSQUnivariateSpline(x, y, knots, k = 3)
    xs = np.linspace(np.min(x), np.max(x), 1000)
    regression_curve = cubic_spline(xs)
    xs_array = np.expand_dims(xs, axis = 1)
    regression_curve_array = np.expand_dims(regression_curve, axis = 1)
    regression_curve_array_for_df = np.hstack((xs_array, regression_curve_array))

    regression_curve_df = pd.DataFrame(regression_curve_array_for_df, columns = ['year', 'energy'])

    plt.title('Cubic Spline (Prediction)')
    sns.lineplot(data = regression_curve_df, x = 'year', y = 'energy')
    sns.scatterplot(data = years_df, x = 'year', y = 'energy', color = ['red'])
    plt.savefig(fname = 'figure_2.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    ## ---------------------------------------------------------
    ## Cubic Spline End\
    ## ---------------------------------------------------------


    ## ---------------------------------------------------------
    ## B-Spline
    ## ---------------------------------------------------------
    list_of_mse = []
    for num_of_knots in range(6, 16):
        knots = np.linspace(np.min(x), np.max(x), num_of_knots)[1: -1]
        kwargs = {
            't': knots,
            's': 0,
            'k': 3
        }
        mse = leave_one_out_cross_validation_b_spline(x, y, splrep, BSpline, kwargs)
        list_of_mse.append(mse)

    knot_counts = np.expand_dims(np.arange(6, 16), axis = 1)
    mse_array = np.expand_dims(np.array(list_of_mse), axis = 1)
    mse_per_knot = np.hstack((knot_counts, mse_array))
    mse_per_knot_df = pd.DataFrame(mse_per_knot, columns = ['knots', 'mse'])

    plt.title('B-Spline (LOOCV)')
    sns.lineplot(data = mse_per_knot_df, x = 'knots', y = 'mse')
    plt.savefig(fname = 'figure_3.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    number_of_knots = np.argmin(mse_array) + 6
    knots = np.linspace(np.min(x), np.max(x), number_of_knots)[1: -1]
    b_spline_rep = splrep(x, y, s = 0, t = knots, k = 3)
    b_spline = BSpline(*b_spline_rep)

    xs = np.linspace(np.min(x), np.max(x), 1000)
    regression_curve = b_spline(xs)
    xs_array = np.expand_dims(xs, axis = 1)
    regression_curve_array = np.expand_dims(regression_curve, axis = 1)

    regression_curve_array_for_df = np.hstack((xs_array, regression_curve_array))
    regression_curve_df = pd.DataFrame(regression_curve_array_for_df, columns = ['year', 'energy'])

    plt.title('B-Spline (Prediction)')
    sns.lineplot(data = regression_curve_df, x = 'year', y = 'energy')
    sns.scatterplot(data = years_df, x = 'year', y = 'energy', color = ['red'])
    plt.savefig(fname = 'figure_4.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    ## ---------------------------------------------------------
    ## B-Spline End\
    ## ---------------------------------------------------------

    ## ---------------------------------------------------------
    ## Smoothing Spline
    ## ---------------------------------------------------------
    list_of_mse = []
    lambda_values = np.arange(0, 1e-3, 1e-6)
    for lambda_val in lambda_values:
        kwargs = {
            'smooth': lambda_val
        }
        mse = leave_one_out_cross_validation_c_spline(x, y, csaps, kwargs)
        list_of_mse.append(mse)

    lambda_val_array = np.expand_dims(lambda_values, axis = 1)
    mse_array = np.expand_dims(np.array(list_of_mse), axis = 1)
    mse_per_lambda_val = np.hstack((lambda_val_array, mse_array))
    mse_per_lambda_val_df = pd.DataFrame(mse_per_lambda_val, columns = ['lambda_val', 'mse'])

    plt.title('Smoothing Spline (LOOCV)')
    sns.lineplot(data = mse_per_lambda_val_df, x = 'lambda_val', y = 'mse')
    plt.savefig(fname = 'figure_5.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    best_lambda_val_value = lambda_values[np.argmin(mse_array)]
    cubic_spline = csaps(x, y, smooth = best_lambda_val_value)
    xs = np.linspace(np.min(x), np.max(x), 1000)
    regression_curve = cubic_spline(xs)
    xs_array = np.expand_dims(xs, axis = 1)
    regression_curve_array = np.expand_dims(regression_curve, axis = 1)
    regression_curve_array_for_df = np.hstack((xs_array, regression_curve_array))

    regression_curve_df = pd.DataFrame(regression_curve_array_for_df, columns = ['year', 'energy'])
    plt.title('Smoothing Spline (Prediction)')
    sns.lineplot(data = regression_curve_df, x = 'year', y = 'energy')
    sns.scatterplot(data = years_df, x = 'year', y = 'energy', color = ['red'])
    plt.savefig(fname = 'figure_6.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    ## ---------------------------------------------------------
    ## Smoothing Spline End\
    ## ---------------------------------------------------------

    ## ---------------------------------------------------------
    ## Gaussian Kernel Regression
    ## ---------------------------------------------------------
    list_of_mse = []
    lambda_val = np.arange(0.2, 1.5, 1e-2)
    for lam in lambda_val:
        kwargs = {
            'kernel': RBF(lam),
            'random_state': 1234
        }
        mse = leave_one_out_cross_validation_gaussian(x, y, GaussianProcessRegressor, kwargs)
        list_of_mse.append(mse)

    lambda_array = np.expand_dims(lambda_val, axis = 1)
    mse_array = np.expand_dims(np.array(list_of_mse), axis = 1)
    mse_per_lambda = np.hstack((lambda_array, mse_array))
    mse_per_lambda_df = pd.DataFrame(mse_per_lambda, columns = ['lambda', 'mse'])

    plt.title('Gaussian Kernel Regression (LOOCV)')
    sns.lineplot(data = mse_per_lambda_df, x = 'lambda', y = 'mse')
    plt.savefig(fname = 'figure_7.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    best_kernel = lambda_array[np.argmin(mse_array)]
    gaussian = GaussianProcessRegressor(RBF(best_kernel), random_state = 1234).fit(x.reshape(-1, 1), y.reshape(-1, 1))
    xs_array = np.expand_dims(np.linspace(np.min(x), np.max(x), 1000), axis = 1)
    regression_curve_array = gaussian.predict(xs_array)
    regression_curve_array_for_df = np.hstack((xs_array, regression_curve_array))

    regression_curve_df = pd.DataFrame(regression_curve_array_for_df, columns = ['year', 'energy'])
    plt.title('Gaussian Kernel Regression (Prediction)')
    sns.lineplot(data = regression_curve_df, x = 'year', y = 'energy')
    sns.scatterplot(data = years_df, x = 'year', y = 'energy', color = ['red'])
    plt.savefig(fname = 'figure_8.png', dpi = 80)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    ## ---------------------------------------------------------
    ## Gaussian Kernel Regression End\
    ## ---------------------------------------------------------