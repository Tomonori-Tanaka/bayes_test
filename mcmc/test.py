import numpy as np
import pymc3 as pm
from pymc3 import Model
import arviz as az
from matplotlib import pyplot as plt
from sklearn import preprocessing


def main():
    ss = preprocessing.StandardScaler()
    data = np.loadtxt("ofm_eform_summary.txt", dtype='float')
    size = 1000
    #x = data[:, :1].reshape(1, len(data))
    #x = x[0]
    x = data[:, :1]
    x_s = ss.fit_transform(x)
    x_s_1column = x_s.reshape(1, len(x_s))
    x_s_1column = x_s_1column[0]
    print(x_s_1column)

    #y = data[:, 1:].reshape(1, len(data))
    #y = y[0]
    y = data[:, 1:]
    y_s = ss.fit_transform(y)
    y_s_1column = y_s.reshape(1, len(y_s))
    y_s_1column = y_s_1column[0]
    print(y_s_1column)

    with Model() as model:
        var = 20
        coef_0 = pm.Normal('coef_0', mu=0, sigma=var)
        coef_1 = pm.Normal('coef_1', mu=0, sigma=var)
        coef_2 = pm.Normal('coef_2', mu=0, sigma=var)
        # coef_3 = pm.Normal('coef_3', mu=0, sigma=var)

        sigma = pm.Uniform('sigma', lower=0, upper=1)

        # y_pred = coef_0 + coef_1 * x + coef_2 * x ** 2 + coef_3 * x ** 3
        # y_pred = coef_0 + coef_1 * x + coef_2 * x ** 2
        y_pred = coef_0 + coef_1 * x_s + coef_2 * x_s ** 2

        Ylikelihood = pm.Normal('Ylikelihood', mu=y_pred, sigma=sigma, observed=y_s)

    map_estimate = pm.find_MAP(model=model)
    print(map_estimate)
    trace = pm.sample(10000, model=model, start=map_estimate)
    print(pm.summary(trace).round(4))
    with model:
        pass
        # az.plot_trace(trace);
        # plt.show()
    plt.scatter(x_s_1column, y_s_1column, label="data")
    plt.show()

if __name__ == '__main__':
    main()
