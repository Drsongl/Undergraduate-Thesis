import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy.stats import chi2


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[~np.isnan(actual)].flatten()
    actual = actual[~np.isnan(actual)].flatten()
    return mean_squared_error(pred, actual)


def var_params(data):
    model = VAR(data)
    res = model.fit(2)

    # when the parameter is insignificant, let's set it as 0
    r = res.params.copy()
    rp = res.pvalues.copy()
    r[rp > 0.05] = 0

    # here A1, A2 is the transpose of mathematical A1, A2.
    A1 = r[1:4, :]
    A2 = r[4:, :]

    # in case all parameters in A1 is insignificant.
    for i in range(3):
        if np.any(A1[:, i]):
            A1[i, i] = 1

    return A1, A2


def model_diagnosis(pred_matrix, actual_matrix, title_add='', df=167*1312, family='gaussian'):
    pred = pred_matrix[~np.isnan(actual_matrix)].flatten()
    actual = actual_matrix[~np.isnan(actual_matrix)].flatten()

    if family == 'gaussian':
        theta = np.sqrt(np.sum((actual - pred) ** 2) / df)
        y = (actual - pred) / theta
    elif family == 'poisson':
        y = np.sqrt(np.abs(2*(actual*np.log((actual+0.1)/(pred+0.1))+actual-pred))) * np.sign(actual-pred)
    else:
        raise ValueError('Family parameter only support Gaussian and Poisson')

    if family == 'poisson':
        total = np.sum(2*(actual*np.log((actual+0.1)/(pred+0.1))+actual-pred))
    else:
        total = np.sum(y**2)

    p = 1 - chi2.cdf(total, df)

    plt.figure(figsize=[10, 6])
    plt.title(title_add + '\nresiduals plot'+'\nDeviance=%d, df=%d, p-value=%.3f' % (total, df, p))
    plt.scatter(actual, y, label='Deviance Residuals')
    plt.hlines(y=0, xmax=max(actual), xmin=0, linestyles='dashed')
    plt.xlabel('y')
    plt.ylabel('Deviance Residuals')
    plt.legend()
    plt.xlim((-10, 160))
    plt.ylim((-15, 25))
    plt.savefig(r'D:\thesis\figures\diagnosis\%s.png' % title_add)
    plt.show()


def test_pearson_residuals(pred_matrix, actual_matrix, title_add=''):
    pred = pred_matrix[~np.isnan(actual_matrix)].flatten()
    actual = actual_matrix[~np.isnan(actual_matrix)].flatten()

    plt.figure(figsize=[10, 6])
    plt.title('Test Set Pearson Residuals\n' + title_add)
    plt.scatter(actual, (actual - pred)/np.sqrt(pred+0.1), label='pearson residuals')
    plt.hlines(y=0, xmax=max(actual), xmin=0, linestyles='dashed')
    plt.xlabel('y')
    plt.ylabel('delta')
    plt.legend()

    plt.savefig(r'D:\thesis\figures\residuals\%s.png' % title_add)
    plt.show()


def plot_inference_delta(pred_matrix, actual_matrix, title_add=''):
    pred = pred_matrix[~np.isnan(actual_matrix)].flatten()
    actual = actual_matrix[~np.isnan(actual_matrix)].flatten()

    num = pred-actual

    plt.figure(figsize=[10, 6])
    plt.title('Real Demand Inference\n' + title_add + '\nInference Data - Observed Data')
    plt.scatter(actual, pred - actual, label='real demand inference - observed data')
    plt.hlines(y=0, xmax=max(actual), xmin=0, linestyles='dashed')
    plt.xlabel('y')
    plt.ylabel('delta')
    plt.legend()

    plt.savefig(r'D:\thesis\figures\inference\%s.png' % title_add)
    plt.show()


class Contingency_table:
    def __init__(self,
                 count=np.zeros(shape=(1, 1)),
                 test=np.zeros(shape=(1, 1)),
                 learning='sgd',
                 geo_reg=0.0,
                 temp_reg=0.0,
                 process=False,
                 verbose=False):

        self.count = count
        self.test = test
        self.n_temp, self.n_geo = count.shape
        self.geo_reg = geo_reg
        self.temp_reg = temp_reg
        self.learning = learning
        self.train_mse = []
        self.test_mse = []
        if self.learning == 'sgd':
            index_matrix = np.isnan(self.count)
            index_matrix = ~index_matrix
            self.sample_row, self.sample_col = index_matrix.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose
        self.process = process

    def fit(self, n_iter=100, learning_rate=0.001):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors

        self.learning_rate = learning_rate
        self.temp_bias = np.zeros(self.n_temp-1)
        self.geo_bias = np.zeros(self.n_geo-1)
        self.global_bias = np.mean(self.count[~np.isnan(self.count)])
        self.partial_train(n_iter)


    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\t\tcurrent iteration: {}'.format(ctr))

            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            self.sgd()

            if self.process:
                self.train_mse.append(get_mse(self.predict_all(), self.count))
                self.test_mse.append(get_mse(self.predict_all(), self.test))
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            t = self.sample_row[idx]
            g = self.sample_col[idx]
            prediction = self.predict(t, g)
            e = (self.count[t, g] - prediction)  # error

            # Update biases
            if (t == self.n_temp - 1) or (g == self.n_geo - 1):
                self.global_bias += self.learning_rate * e
            else:
                self.temp_bias[t] += self.learning_rate * (e - self.temp_reg * self.temp_bias[t])
                self.geo_bias[g] += self.learning_rate * (e - self.geo_reg * self.geo_bias[g])

    def predict(self, t, g):
        """ Single temporal and spacial prediction."""

        # log(mu) = mu0 + t_i + g_j
        # mu = exp( mu0 + t_i + g_j )

        # 设置预测值的范围，防止exp爆炸造成梯度爆炸。
        # exp（10）=22k，接近一个时段所有地点的汇总，
        # exp(9)=8100, 学习率为0.001时，可以最大调整8倍，
        # exp(8)=2980，学习率为0.001时，最大可以调整3倍，
        # 而一个时间一个地点，最大值为169

        prediction = self.global_bias
        if t < (self.n_temp - 1):
            prediction += self.temp_bias[t]
        if g < (self.n_geo - 1):
            prediction += self.geo_bias[g]

        if prediction > 9:
            prediction = 9
        prediction = np.exp(prediction)
        return prediction

    def predict_all(self):
        """ Predict count for every user and item."""
        predictions = np.zeros((self.n_temp, self.n_geo))
        for t in range(self.n_temp):
            for g in range(self.n_geo):
                predictions[t, g] = self.predict(t, g)

        return predictions

    def plot_learning_curve(self, title_add='', MSE=''):
        if self.process:
            min_train_mse_index = np.argmin(self.train_mse)
            min_test_mse_index = np.argmin(self.test_mse)
            plt.title('Contingency Table: Training & Test MSE\n' + title_add + MSE)

            plt.plot(self.train_mse,
                     label='training_mse, min=[%d, %.3f]' % (min_train_mse_index, self.train_mse[min_train_mse_index]))
            plt.plot(min_train_mse_index, self.train_mse[min_train_mse_index], 'ko')

            plt.plot(self.test_mse,
                     label='test_mse, min=[%d, %.3f]' % (min_test_mse_index, self.test_mse[min_test_mse_index]))
            plt.plot(min_test_mse_index, self.test_mse[min_test_mse_index], 'ko')

            plt.xlabel('iteration')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(r'D:\thesis\figures\MSE\CT_' + title_add + '.png')
            plt.show()
        else:
            print('Process MSE is not recorded\nIf you want to plot the learning curve, '
                  'make sure the process parameter is True')


class ExplicitMF:
    def __init__(self,
                 rating=np.zeros(shape=(1, 1)),
                 test=np.zeros(shape=(1, 1)),
                 n_factors=40,
                 learning='sgd',
                 item_fact_reg=0.0,
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 process=False,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        rating matrix which is ~ user x item

        Params
        ======
        rating : (ndarray)
            User x Item matrix with corresponding rating

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model
        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        item_fact_reg : (float)
            Regularization term for item latent factors

        user_fact_reg : (float)
            Regularization term for user latent factors

        item_bias_reg : (float)
            Regularization term for item biases

        user_bias_reg : (float)
            Regularization term for user biases

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.rating = rating
        self.test = test
        self.n_users, self.n_items = rating.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.train_mse = []
        self.test_mse = []
        self.learning = learning
        self.process = process
        if self.learning == 'sgd':
            index_matrix = np.isnan(self.rating)
            index_matrix = ~index_matrix
            self.sample_row, self.sample_col = index_matrix.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 rating,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             rating[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             rating[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def fit(self, n_iter=100, learning_rate=0.001):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_items, self.n_factors))

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.rating[~np.isnan(self.rating)])
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\t\tcurrent iteration: {}'.format(ctr))
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs,
                                               self.item_vecs,
                                               self.rating,
                                               self.user_fact_reg,
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs,
                                               self.user_vecs,
                                               self.rating,
                                               self.item_fact_reg,
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            if self.process:
                self.train_mse.append(get_mse(self.predict_all(), self.rating))
                self.test_mse.append(get_mse(self.predict_all(), self.test))
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.rating[u, i] - prediction)  # error

            # Update biases
            self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * (e - self.item_bias_reg * self.item_bias[i])

            # Update latent factors
            self.user_vecs[u, :] += self.learning_rate * (
                        e * self.item_vecs[i, :] - self.user_fact_reg * self.user_vecs[u, :])
            self.item_vecs[i, :] += self.learning_rate * (
                        e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i, :])

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction

    def predict_all(self):
        """ Predict rating for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0], self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)

        return predictions

    def plot_learning_curve(self, title_add='', MSE=''):
        if self.process:
            min_train_mse_index = np.argmin(self.train_mse)
            min_test_mse_index = np.argmin(self.test_mse)
            plt.title('MF: Training & Test MSE\n' + title_add + MSE)

            plt.plot(self.train_mse,
                     label='training_mse, min=[%d, %.3f]' % (min_train_mse_index, self.train_mse[min_train_mse_index]))
            plt.plot(min_train_mse_index, self.train_mse[min_train_mse_index], 'ko')

            plt.plot(self.test_mse,
                     label='test_mse, min=[%d, %.3f]' % (min_test_mse_index, self.test_mse[min_test_mse_index]))
            plt.plot(min_test_mse_index, self.test_mse[min_test_mse_index], 'ko')

            plt.xlabel('iteration')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(r'D:\thesis\figures\MSE\MF_'+title_add+'.png')
            plt.show()
        else:
            print('Process MSE is not recorded\nIf you want to plot the learning curve, '
                  'make sure the process parameter is True')


class MF_VAR:
    def __init__(self,
                 count=np.zeros(shape=(1, 1)),
                 test=np.zeros(shape=(1, 1)),
                 n_factors=40,
                 learning='sgd',
                 geo_reg=0.0,
                 temp_reg=0.0,
                 geo_bias_reg=0.0,
                 temp_bias_reg=0.0,
                 var_reg=0.0,
                 A1=1.04 * np.eye(3),
                 A2=-0.38 * np.eye(3),
                 process=False,
                 verbose=False):

        self.count = count
        self.test = test
        self.n_temp, self.n_geo = count.shape
        self.n_factors = n_factors
        self.geo_reg = geo_reg
        self.temp_reg = temp_reg
        self.geo_bias_reg = geo_bias_reg
        self.temp_bias_reg = temp_bias_reg
        self.var_reg = var_reg
        self.train_mse = []
        self.test_mse = []
        self.learning = learning
        self.process = process
        self.A1 = A1
        self.A2 = A2

        self.geo_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_geo, self.n_factors))
        self.temp_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_temp, self.n_factors))

        self.temp_bias = np.zeros(self.n_temp)
        self.geo_bias = np.zeros(self.n_geo)
        self.global_bias = np.mean(self.count[~np.isnan(self.count)])

        if self.learning == 'sgd':
            index_matrix = np.isnan(self.count)
            index_matrix = ~index_matrix
            self.sample_row, self.sample_col = index_matrix.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 count,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'temp':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             count[u, :].dot(fixed_vecs))
        elif type == 'geo':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             count[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def fit(self, n_iter=100, learning_rate=0.001):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0:

                # update A for VAR model in every 10-iteration
                diff_temp = pd.DataFrame(self.temp_vecs).diff(24).values[24:, :]
                self.A1, self.A2 = var_params(diff_temp)

                if self._v:
                    print('\t\tcurrent iteration: {}'.format(ctr))
                    print('A1=', self.A1, '\nA2=', self.A2)

            if self.learning == 'als':
                self.temp_vecs = self.als_step(self.temp_vecs,
                                               self.geo_vecs,
                                               self.count,
                                               self.temp_reg,
                                               type='temp')
                self.geo_vecs = self.als_step(self.geo_vecs,
                                               self.temp_vecs,
                                               self.count,
                                               self.geo_reg,
                                               type='geo')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            if self.process:
                self.train_mse.append(get_mse(self.predict_all(), self.count))
                self.test_mse.append(get_mse(self.predict_all(), self.test))
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            t = self.sample_row[idx]
            g = self.sample_col[idx]
            prediction = self.predict(t, g)
            e = (self.count[t, g] - prediction)  # error

            # Update biases
            self.temp_bias[t] += self.learning_rate * (e - self.temp_bias_reg * self.temp_bias[t])
            self.geo_bias[g] += self.learning_rate * (e - self.geo_bias_reg * self.geo_bias[g])

            # Update latent factors
            self.temp_vecs[t, :] += self.learning_rate * (
                    e * self.geo_vecs[g, :] - self.temp_reg * self.temp_vecs[t, :] - self.var_reg * self.d_var(t))
            self.geo_vecs[g, :] += self.learning_rate * (
                        e * self.temp_vecs[t, :] - self.geo_reg * self.geo_vecs[g, :])

        # in my problem, i have a VAR(2) model for 24th difference of original data.

    def d_var(self, i):
        if i == 0:
            deviation = self.partial_d_var(i, lag=6)
            return deviation
        elif i == 1:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5)
            return deviation
        elif 2 <= i <= 23:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4)
            return deviation
        elif i == 24:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4) \
                        + self.partial_d_var(i, lag=3)
            return deviation
        elif i == 25:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4) \
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2)
            return deviation
        elif 26 <= i <= (self.n_temp - 27):
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4) \
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 26):
            deviation = self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4) \
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 25):
            deviation = self.partial_d_var(i, lag=4) \
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif (self.n_temp - 24) <= i <= (self.n_temp - 3):
            deviation = self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 2):
            deviation = self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 1):
            deviation = self.partial_d_var(i, lag=1)
            return deviation
        else:
            try:
                raise ValueError('i is larger than n_temp!')
            except ValueError:
                print('An exception flew by!')
                raise

    def partial_d_var(self, i, lag):
        if lag == 1:
            d = self.temp_vecs[i, :] - np.dot(self.temp_vecs[i - 1, :], self.A1) - np.dot(self.temp_vecs[i - 2, :],
                                                                                          self.A2) \
                - self.temp_vecs[i - 24, :] + np.dot(self.temp_vecs[i - 25, :], self.A1) + np.dot(
                self.temp_vecs[i - 26, :], self.A2)
        elif lag == 2:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A1), self.A1.T) - np.dot(self.temp_vecs[i + 1, :], self.A1.T) \
                + np.dot(np.dot(self.temp_vecs[i - 1, :], self.A2), self.A1.T) - np.dot(
                np.dot(self.temp_vecs[i - 24, :], self.A1), self.A1.T) \
                + np.dot(self.temp_vecs[i - 23, :], self.A1.T) - np.dot(np.dot(self.temp_vecs[i - 25, :], self.A2),
                                                                        self.A1.T)
        elif lag == 3:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A2), self.A2.T) - np.dot(self.temp_vecs[i + 2, :], self.A2.T) \
                + np.dot(np.dot(self.temp_vecs[i + 1, :], self.A1), self.A2.T) - np.dot(
                np.dot(self.temp_vecs[i - 24, :], self.A2), self.A2.T) \
                + np.dot(self.temp_vecs[i - 22, :], self.A2.T) - np.dot(np.dot(self.temp_vecs[i - 23, :], self.A1),
                                                                        self.A2.T)
        elif lag == 4:
            d = self.temp_vecs[i, :] - np.dot(self.temp_vecs[i - 1, :], self.A1) - np.dot(self.temp_vecs[i - 2, :],
                                                                                          self.A2) \
                - self.temp_vecs[i + 24, :] + np.dot(self.temp_vecs[i + 23, :], self.A1) + np.dot(
                self.temp_vecs[i + 22, :], self.A2)
        elif lag == 5:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A1), self.A1.T) - np.dot(self.temp_vecs[i + 1, :], self.A1.T) \
                + np.dot(np.dot(self.temp_vecs[i - 1, :], self.A2), self.A1.T) - np.dot(
                np.dot(self.temp_vecs[i + 24, :], self.A1), self.A1.T) \
                + np.dot(self.temp_vecs[i + 25, :], self.A1.T) - np.dot(np.dot(self.temp_vecs[i + 23, :], self.A2),
                                                                        self.A1.T)
        elif lag == 6:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A2), self.A2.T) - np.dot(self.temp_vecs[i + 2, :], self.A2.T) \
                + np.dot(np.dot(self.temp_vecs[i + 1, :], self.A1), self.A2.T) - np.dot(
                np.dot(self.temp_vecs[i + 24, :], self.A2), self.A2.T) \
                + np.dot(self.temp_vecs[i + 26, :], self.A2.T) - np.dot(np.dot(self.temp_vecs[i + 25, :], self.A1),
                                                                        self.A2.T)
        else:
            try:
                raise ValueError('partial deviation of VAR, only allows 1,2,3,4,5,6 as lag')
            except ValueError:
                print('An exception flew by!')
                raise
        return d

    def predict(self, t, g):
        """ Single temp and geo prediction."""
        if self.learning == 'als':
            return self.temp_vecs[t, :].dot(self.geo_vecs[g, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.temp_bias[t] + self.geo_bias[g]
            prediction += self.temp_vecs[t, :].dot(self.geo_vecs[g, :].T)
            return prediction

    def predict_all(self):
        """ Predict count for every temp and geo."""
        predictions = np.zeros((self.temp_vecs.shape[0], self.geo_vecs.shape[0]))
        for t in range(self.temp_vecs.shape[0]):
            for g in range(self.geo_vecs.shape[0]):
                predictions[t, g] = self.predict(t, g)

        return predictions

    def plot_learning_curve(self, title_add='', MSE=''):
        if self.process:
            min_train_mse_index = np.argmin(self.train_mse)
            min_test_mse_index = np.argmin(self.test_mse)
            plt.title('MF VAR: Training & Test MSE\n' + title_add + MSE)

            plt.plot(self.train_mse,
                     label='training_mse, min=[%d, %.3f]' % (min_train_mse_index, self.train_mse[min_train_mse_index]))
            plt.plot(min_train_mse_index, self.train_mse[min_train_mse_index], 'ko')

            plt.plot(self.test_mse,
                     label='test_mse, min=[%d, %.3f]' % (min_test_mse_index, self.test_mse[min_test_mse_index]))
            plt.plot(min_test_mse_index, self.test_mse[min_test_mse_index], 'ko')

            plt.xlabel('iteration')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(r'D:\thesis\figures\MSE\MF_VAR_'+title_add+'.png')
            plt.show()
        else:
            print('Process MSE is not recorded\nIf you want to plot the learning curve, '
                  'make sure the process parameter is True')


class PoissonMF:
    def __init__(self,
                 count=np.zeros(shape=(1, 1)),
                 test=np.zeros(shape=(1, 1)),
                 n_factors=40,
                 learning='sgd',
                 geo_reg=0.0,
                 temp_reg=0.0,
                 geo_bias_reg=0.0,
                 temp_bias_reg=0.0,
                 process=False,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        count matrix which is ~ temporal x spatial

        Params
        ======
        count : (ndarray)
            Temporal x Spatial matrix with corresponding count

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model

        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        geo_reg : (float)
            Regularization term for spatial latent factors

        temp_reg : (float)
            Regularization term for temporal latent factors

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.count = count
        self.test = test
        self.n_temp, self.n_geo = count.shape
        self.n_factors = n_factors
        self.geo_reg = geo_reg
        self.temp_reg = temp_reg
        self.geo_bias_reg = geo_bias_reg
        self.temp_bias_reg = temp_bias_reg
        self.learning = learning
        self.process = process
        self.train_mse = []
        self.test_mse = []
        if self.learning == 'sgd':
            index_matrix = np.isnan(self.count)
            index_matrix = ~index_matrix
            self.sample_row, self.sample_col = index_matrix.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 count,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             count[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             count[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def fit(self, n_iter=100, learning_rate=0.001):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.temp_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_temp, self.n_factors))
        self.geo_vecs = np.random.normal(scale=1. / self.n_factors,
                                         size=(self.n_geo, self.n_factors))

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.temp_bias = np.zeros(self.n_temp)
            self.geo_bias = np.zeros(self.n_geo)
            self.global_bias = np.mean(self.count[~np.isnan(self.count)])
            self.partial_train(n_iter)


    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\t\tcurrent iteration: {}'.format(ctr))
            if self.learning == 'als':
                self.temp_vecs = self.als_step(self.temp_vecs,
                                               self.geo_vecs,
                                               self.count,
                                               self.temp_reg,
                                               type='user')
                self.geo_vecs = self.als_step(self.geo_vecs,
                                              self.temp_vecs,
                                              self.count,
                                              self.geo_reg,
                                              type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            if self.process:
                self.train_mse.append(get_mse(self.predict_all(), self.count))
                self.test_mse.append(get_mse(self.predict_all(), self.test))
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            t = self.sample_row[idx]
            g = self.sample_col[idx]
            prediction = self.predict(t, g)
            e = (self.count[t, g] - prediction)  # error

            # Update biases
            self.temp_bias[t] += self.learning_rate * (e - self.temp_bias_reg * self.temp_bias[t])
            self.geo_bias[g] += self.learning_rate * (e - self.geo_bias_reg * self.geo_bias[g])

            # Update latent factors
            self.temp_vecs[t, :] += self.learning_rate * (
                        e * self.geo_vecs[g, :] - self.temp_reg * self.temp_vecs[t, :])
            self.geo_vecs[g, :] += self.learning_rate * (
                    e * self.temp_vecs[t, :] - self.geo_reg * self.geo_vecs[g, :])

            # # add regulation for biases and latent factors
            # if self.temp_bias[t] > 1e4:
            #     self.temp_bias[t] = 1e4
            # if self.temp_bias[t] < -1e4:
            #     self.temp_bias[t] = -1e4
            # if self.geo_bias[g] < -1e4:
            #     self.geo_bias[g] = -1e4
            # if self.geo_bias[g] > 1e4:
            #     self.geo_bias[g] = 1e4
            #
            # self.temp_vecs[t, :][self.temp_vecs[t, :] > 1e4] = 1e4
            # self.geo_vecs[g, :][self.geo_vecs[g, :] > 1e4] = 1e4
            # self.temp_vecs[t, :][self.temp_vecs[t, :] < -1e4] = -1e4
            # self.geo_vecs[g, :][self.geo_vecs[g, :] < -1e4] = -1e4


    def predict(self, t, g):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.temp_vecs[t, :].dot(self.geo_vecs[g, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.temp_bias[t] + self.geo_bias[g]
            prediction += self.temp_vecs[t, :].dot(self.geo_vecs[g, :].T)
            # log(mu) = mu0 + t_i + g_j + T_i.T * G_j
            # mu = exp( mu0 + t_i + g_j + T_i.T * G_j )

            # 设置预测值的范围，防止exp爆炸造成梯度爆炸。
            # exp（10）=22k，接近一个时段所有地点的汇总，
            # exp(9)=8100, 学习率为0.001时，可以最大调整8倍，
            # exp(8)=2980，学习率为0.001时，最大可以调整3倍，
            # 而一个时间一个地点，最大值为169
            if prediction > 9:
                prediction = 9
            prediction = np.exp(prediction)
            return prediction

    def predict_all(self):
        """ Predict count for every user and item."""
        predictions = np.zeros((self.temp_vecs.shape[0], self.geo_vecs.shape[0]))
        for t in range(self.temp_vecs.shape[0]):
            for g in range(self.geo_vecs.shape[0]):
                predictions[t, g] = self.predict(t, g)

        return predictions

    def plot_learning_curve(self, title_add='', MSE=''):
        if self.process:
            min_train_mse_index = np.argmin(self.train_mse)
            min_test_mse_index = np.argmin(self.test_mse)
            plt.title('PMF: Training & Test MSE\n' + title_add + MSE)

            plt.plot(self.train_mse,
                     label='training_mse, min=[%d, %.3f]' % (min_train_mse_index, self.train_mse[min_train_mse_index]))
            plt.plot(min_train_mse_index, self.train_mse[min_train_mse_index], 'ko')

            plt.plot(self.test_mse,
                     label='test_mse, min=[%d, %.3f]' % (min_test_mse_index, self.test_mse[min_test_mse_index]))
            plt.plot(min_test_mse_index, self.test_mse[min_test_mse_index], 'ko')

            plt.xlabel('iteration')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(r'D:\thesis\figures\MSE\PMF_'+title_add+'.png')
            plt.show()
        else:
            print('Process MSE is not recorded\nIf you want to plot the learning curve, '
                  'make sure the process parameter is True')


class PoissonMF_VAR:
    def __init__(self,
                 count=np.zeros(shape=(1, 1)),
                 test=np.zeros(shape=(1, 1)),
                 n_factors=40,
                 learning='sgd',
                 geo_reg=0.0,
                 temp_reg=0.0,
                 geo_bias_reg=0.0,
                 temp_bias_reg=0.0,
                 var_reg=0.0,
                 A1 = 1.04 * np.eye(3),
                 A2 = -0.38 * np.eye(3),
                 process=False,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        count matrix which is ~ temporal x spatial

        Params
        ======
        count : (ndarray)
            Temporal x Spatial matrix with corresponding count

        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model

        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.

        geo_reg : (float)
            Regularization term for spatial latent factors

        temp_reg : (float)
            Regularization term for temporal latent factors

        verbose : (bool)
            Whether or not to printout training progress
        """

        self.count = count
        self.test = test
        self.n_temp, self.n_geo = count.shape
        self.n_factors = n_factors
        self.geo_reg = geo_reg
        self.temp_reg = temp_reg
        self.geo_bias_reg = geo_bias_reg
        self.temp_bias_reg = temp_bias_reg
        self.var_reg = var_reg
        self.learning = learning
        self.process = process
        self.A1 = A1
        self.A2 = A2

        self.train_mse = []
        self.test_mse = []
        if self.learning == 'sgd':
            index_matrix = np.isnan(self.count)
            index_matrix = ~index_matrix
            self.sample_row, self.sample_col = index_matrix.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 count,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI),
                                             count[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda

            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI),
                                             count[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def fit(self, n_iter=100, learning_rate=0.001):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.temp_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_temp, self.n_factors))
        self.geo_vecs = np.random.normal(scale=1. / self.n_factors,
                                         size=(self.n_geo, self.n_factors))

        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.temp_bias = np.zeros(self.n_temp)
            self.geo_bias = np.zeros(self.n_geo)
            self.global_bias = np.mean(self.count[~np.isnan(self.count)])
            self.partial_train(n_iter)


    def partial_train(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0:
                # update A for VAR model in every 10-iteration
                diff_temp = pd.DataFrame(self.temp_vecs).diff(24).values[24: , :]
                self.A1, self.A2 = var_params(diff_temp)

                if self._v:
                    print('\t\tcurrent iteration: {}'.format(ctr))
                    print('A1=', self.A1, '\nA2=', self.A2)

            if self.learning == 'als':
                self.temp_vecs = self.als_step(self.temp_vecs,
                                               self.geo_vecs,
                                               self.count,
                                               self.temp_reg,
                                               type='user')
                self.geo_vecs = self.als_step(self.geo_vecs,
                                              self.temp_vecs,
                                              self.count,
                                              self.geo_reg,
                                              type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()

            if self.process:
                self.train_mse.append(get_mse(pred=self.predict_all(), actual=self.count))
                self.test_mse.append(get_mse(pred=self.predict_all(), actual=self.test))

            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            t = self.sample_row[idx]
            g = self.sample_col[idx]
            prediction = self.predict(t, g)
            e = (self.count[t, g] - prediction)  # error

            # Update biases
            self.temp_bias[t] += self.learning_rate * (e - self.temp_bias_reg * self.temp_bias[t])
            self.geo_bias[g] += self.learning_rate * (e - self.geo_bias_reg * self.geo_bias[g])

            # Update latent factors

            self.geo_vecs[g, :] += self.learning_rate * (
                    e * self.temp_vecs[t, :] - self.geo_reg * self.geo_vecs[g, :])

            self.temp_vecs[t, :] += self.learning_rate * (
                    e * self.geo_vecs[g, :] - self.temp_reg * self.temp_vecs[t, :] - self.var_reg * self.d_var(t))

    # in my problem, i have a VAR(2) model for 24th difference of original data.
    def d_var(self, i):
        if i == 0:
            deviation = self.partial_d_var(i, lag=6)
            return deviation
        elif i == 1:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5)
            return deviation
        elif 2 <= i <= 23:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4)
            return deviation
        elif i == 24:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4)\
                        + self.partial_d_var(i, lag=3)
            return deviation
        elif i == 25:
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4)\
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2)
            return deviation
        elif 26 <= i <= (self.n_temp - 27):
            deviation = self.partial_d_var(i, lag=6) + self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4)\
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 26):
            deviation = self.partial_d_var(i, lag=5) + self.partial_d_var(i, lag=4)\
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 25):
            deviation = self.partial_d_var(i, lag=4) \
                        + self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif (self.n_temp - 24) <= i <= (self.n_temp - 3):
            deviation = self.partial_d_var(i, lag=3) + self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 2):
            deviation = self.partial_d_var(i, lag=2) + self.partial_d_var(i, lag=1)
            return deviation
        elif i == (self.n_temp - 1):
            deviation = self.partial_d_var(i, lag=1)
            return deviation
        else:
            try:
                raise ValueError('i is larger than n_temp!')
            except ValueError:
                print('An exception flew by!')
                raise

    def partial_d_var(self, i, lag):
        if lag == 1:
            d = self.temp_vecs[i, :] - np.dot(self.temp_vecs[i-1, :], self.A1) - np.dot(self.temp_vecs[i-2, :], self.A2) \
                - self.temp_vecs[i-24, :] + np.dot(self.temp_vecs[i-25, :], self.A1) + np.dot(self.temp_vecs[i-26, :], self.A2)
        elif lag == 2:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A1), self.A1.T) - np.dot(self.temp_vecs[i+1, :], self.A1.T) \
                + np.dot(np.dot(self.temp_vecs[i-1, :], self.A2), self.A1.T) - np.dot(np.dot(self.temp_vecs[i-24, :], self.A1), self.A1.T) \
                + np.dot(self.temp_vecs[i-23, :], self.A1.T) - np.dot(np.dot(self.temp_vecs[i-25, :], self.A2), self.A1.T)
        elif lag == 3:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A2), self.A2.T) - np.dot(self.temp_vecs[i+2, :], self.A2.T) \
                + np.dot(np.dot(self.temp_vecs[i+1, :], self.A1), self.A2.T) - np.dot(np.dot(self.temp_vecs[i-24, :], self.A2), self.A2.T)\
                + np.dot(self.temp_vecs[i-22, :], self.A2.T) - np.dot(np.dot(self.temp_vecs[i-23, :], self.A1), self.A2.T)
        elif lag == 4:
            d = self.temp_vecs[i, :] - np.dot(self.temp_vecs[i-1, :], self.A1) - np.dot(self.temp_vecs[i-2, :], self.A2) \
                - self.temp_vecs[i+24, :] + np.dot(self.temp_vecs[i+23, :], self.A1) + np.dot(self.temp_vecs[i+22, :], self.A2)
        elif lag == 5:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A1), self.A1.T) - np.dot(self.temp_vecs[i+1, :], self.A1.T) \
                + np.dot(np.dot(self.temp_vecs[i-1, :], self.A2), self.A1.T) - np.dot(np.dot(self.temp_vecs[i+24, :], self.A1), self.A1.T) \
                + np.dot(self.temp_vecs[i+25, :], self.A1.T) - np.dot(np.dot(self.temp_vecs[i+23, :], self.A2), self.A1.T)
        elif lag == 6:
            d = np.dot(np.dot(self.temp_vecs[i, :], self.A2), self.A2.T) - np.dot(self.temp_vecs[i+2, :], self.A2.T) \
                + np.dot(np.dot(self.temp_vecs[i+1, :], self.A1), self.A2.T) - np.dot(np.dot(self.temp_vecs[i+24, :], self.A2), self.A2.T)\
                + np.dot(self.temp_vecs[i+26, :], self.A2.T) - np.dot(np.dot(self.temp_vecs[i+25, :], self.A1), self.A2.T)
        else:
            try:
                raise ValueError('partial deviation of VAR, only allows 1,2,3,4,5,6 as lag')
            except ValueError:
                print('An exception flew by!')
                raise
        return d

    def predict(self, t, g):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.temp_vecs[t, :].dot(self.geo_vecs[g, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.temp_bias[t] + self.geo_bias[g]
            prediction += self.temp_vecs[t, :].dot(self.geo_vecs[g, :].T)
            # log(mu) = mu0 + t_i + g_j + T_i.T * G_j
            # mu = exp( mu0 + t_i + g_j + T_i.T * G_j )

            # 设置预测值的范围，防止exp爆炸造成梯度爆炸。
            # exp（10）=22k，接近一个时段所有地点的汇总，
            # exp(9)=8100, 学习率为0.001时，可以最大调整8倍，
            # exp(8)=2980，学习率为0.001时，最大可以调整3倍，
            # 而一个时间一个地点，最大值为169
            if prediction > 9:
                prediction = 9
            prediction = np.exp(prediction)
            return prediction

    def predict_all(self):
        """ Predict count for every user and item."""
        predictions = np.zeros((self.temp_vecs.shape[0], self.geo_vecs.shape[0]))
        for t in range(self.temp_vecs.shape[0]):
            for g in range(self.geo_vecs.shape[0]):
                predictions[t, g] = self.predict(t, g)

        return predictions

    def plot_learning_curve(self, title_add='', MSE=''):
        if self.process:
            min_train_mse_index = np.argmin(self.train_mse)
            min_test_mse_index = np.argmin(self.test_mse)
            plt.title('PMF_VAR: Training & Test MSE\n' + title_add + MSE)

            plt.plot(self.train_mse,
                     label='training_mse, min=[%d, %.3f]' % (min_train_mse_index, self.train_mse[min_train_mse_index]))
            plt.plot(min_train_mse_index, self.train_mse[min_train_mse_index], 'ko')

            plt.plot(self.test_mse,
                     label='test_mse, min=[%d, %.3f]' % (min_test_mse_index, self.test_mse[min_test_mse_index]))
            plt.plot(min_test_mse_index, self.test_mse[min_test_mse_index], 'ko')

            plt.xlabel('iteration')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(r'D:\thesis\figures\MSE\PMF_VAR_' + title_add + '.png')
            plt.show()
        else:
            print('Process MSE is not recorded\nIf you want to plot the learning curve, '
                  'make sure the process parameter is True')

