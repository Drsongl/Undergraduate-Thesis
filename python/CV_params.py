from loss import *
import pandas as pd
import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import ParameterSampler
import random
from scipy.stats import uniform

with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0


def generate_sets(data, groups=5):
    index_matrix = np.isnan(data)
    index_matrix = ~index_matrix
    sample_row, _ = index_matrix.nonzero()
    N_list = [i for i in range(len(sample_row))]
    random.Random(666).shuffle(N_list)

    m = int(len(sample_row) / groups) + 1
    group = []
    for i in range(0, len(sample_row), m):
        group.append(N_list[i:i + m])

    return group


def generate_matrix(data, groups=list(), test_order=0, validation_order=None):

    # get matrix index
    index_matrix = np.isnan(data)
    index_matrix = ~index_matrix
    sample_row, sample_col = index_matrix.nonzero()
    N_list = [i for i in range(len(sample_row))]

    # get test matrix
    test_index = groups[test_order]
    _test_index = list(set(N_list) - set(test_index))
    test_matrix = data.copy()
    test_matrix[sample_row[_test_index], sample_col[_test_index]] = np.nan

    # get validation matrix
    if validation_order is not None:
        validation_index = groups[validation_order]
        _validation_index = list(set(N_list) - set(validation_index))
        validation_matrix = data.copy()
        validation_matrix[sample_row[_validation_index], sample_col[_validation_index]] = np.nan
    else:
        validation_matrix = None

    # get train matrix
    train_index = []
    for i in range(len(groups)):
        if (i != test_order) & (i != validation_order):
            train_index.extend(groups[i])

    _train_index = list(set(N_list)-set(train_index))
    train_matrix = data.copy()
    train_matrix[sample_row[_train_index], sample_col[_train_index]] = np.nan

    return test_matrix, validation_matrix, train_matrix


def best_param(MSE =list(), param_list=list(), verbose=False):
    best_index = np.argmin(MSE)
    best_p = param_list[best_index]
    if verbose:
        print('min MSE=',min(MSE), '\ncorresponding parameters:',best_p)
    return best_p


def cross_val(data, cv=5, n_try_params=30, n_iter=50, learning_rate=0.001, param_list=list(), method='MF',
              A1=1.04 * np.eye(3), A2 = -0.38 * np.eye(3), MSE=None, verbose=True):
    '''

    :param data:
    :param cv: test, validation, train, number
    :param param_list: parameter list
    :param method: use what method to factorization the matrix
    :param verbose: to print or not
    :return: list of MSE values
    '''
    if MSE is None:
        MSE = np.ones(shape=(n_try_params, cv - 1)) * 100
    train_test_val_split = generate_sets(data, groups=cv)

    for i in range(1, cv):
        _, validation_matrix, train_matrix = generate_matrix(data, groups=train_test_val_split,
                                                             test_order=0, validation_order=i)

        if verbose:
            print('%d cross validation' % i)

        for j in range(n_try_params):
            params = list(param_list)[j]

            if verbose:
                print('\t%d th parameter' % (j+1))

            if method == 'MF':
                MF_SGD = ExplicitMF(train_matrix, n_factors=3, learning='sgd', verbose=True,
                                      user_bias_reg=params['temp_bias_reg'],
                                      item_bias_reg=params['geo_bias_reg'],
                                      user_fact_reg=params['temp_reg'],
                                      item_fact_reg=params['geo_reg'])
                MF_SGD.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = MF_SGD.predict_all()
                if verbose:
                    print('\ttemp_reg=%.4f\t\tgeo_reg=%.4f\n\ttemp_bias_reg=%.4f\tgeo_bias_reg=%.4f' % (
                        params['temp_reg'], params['geo_reg'], params['temp_bias_reg'], params['geo_bias_reg']))

            elif method == 'MF_VAR':
                MF_var = MF_VAR(train_matrix, validation_matrix, n_factors=3, learning='sgd', verbose=True,
                                            temp_bias_reg=params['temp_bias_reg'],
                                            geo_bias_reg=params['geo_bias_reg'],
                                            temp_reg=params['temp_reg'],
                                            geo_reg=params['geo_reg'],
                                            var_reg=params['var_reg'],
                                            A1=A1, A2=A2)
                MF_var.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = MF_var.predict_all()
                if verbose:
                    print('\ttemp_reg=%.4f\t\tgeo_reg=%.4f\n\ttemp_bias_reg=%.4f\tgeo_bias_reg=%.4f\nvar_reg=%.4f' % (
                        params['temp_reg'], params['geo_reg'], params['temp_bias_reg'], params['geo_bias_reg'],
                        params['var_reg']))

            elif method == 'MF_VAR_single':
                MF_var = MF_VAR(train_matrix, validation_matrix, n_factors=3, learning='sgd', verbose=True,
                                var_reg=params['var_reg'], A1=A1, A2=A2)
                MF_var.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = MF_var.predict_all()
                if verbose:
                    print('\tvar_reg=%.4f' % (params['var_reg']))

            elif method == 'PMF':
                PMF_SGD = PoissonMF(train_matrix, validation_matrix, n_factors=3, learning='sgd', verbose=True,
                                      temp_bias_reg=params['temp_bias_reg'],
                                      geo_bias_reg=params['geo_bias_reg'],
                                      temp_reg=params['temp_reg'],
                                      geo_reg=params['geo_reg'])
                PMF_SGD.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = PMF_SGD.predict_all()
                if verbose:
                    print('\ttemp_reg=%.4f\t\tgeo_reg=%.4f\n\ttemp_bias_reg=%.4f\tgeo_bias_reg=%.4f' % (
                        params['temp_reg'], params['geo_reg'], params['temp_bias_reg'], params['geo_bias_reg']))

            elif method == 'PMF_VAR':
                PMF_var_SGD = PoissonMF_VAR(train_matrix, validation_matrix, n_factors=3, learning='sgd', verbose=True,
                                          temp_bias_reg=params['temp_bias_reg'],
                                          geo_bias_reg=params['geo_bias_reg'],
                                          temp_reg=params['temp_reg'],
                                          geo_reg=params['geo_reg'],
                                          var_reg=params['var_reg'],
                                          A1=A1, A2=A2)
                PMF_var_SGD.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = PMF_var_SGD.predict_all()
                if verbose:
                    print('\ttemp_reg=%.4f\t\tgeo_reg=%.4f\n\ttemp_bias_reg=%.4f\tgeo_bias_reg=%.4f\nvar_reg=%.4f' % (
                        params['temp_reg'], params['geo_reg'], params['temp_bias_reg'], params['geo_bias_reg'],
                        params['var_reg']))

            elif method == 'PMF_VAR_single':
                PMF_var_SGD = PoissonMF_VAR(train_matrix, validation_matrix, n_factors=3, learning='sgd', verbose=True,
                                            var_reg=params['var_reg'], A1=A1, A2=A2)
                PMF_var_SGD.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = PMF_var_SGD.predict_all()
                if verbose:
                    print('\tvar_reg=%.4f' % (params['var_reg']))

            elif method == 'Contingency_table':
                CT = Contingency_table(train_matrix, validation_matrix,
                                       geo_reg=params['geo_reg'], temp_reg=params['temp_reg'])
                CT.fit(n_iter=n_iter, learning_rate=learning_rate)
                pred_matrix = CT.predict_all()
                if verbose:
                    print('\ttemp_reg=%.4f\t\tgeo_reg=%.4f' % (
                        params['temp_reg'], params['geo_reg']))

            else:
                try:
                    raise NameError('Only CT, MF, PMF, and PMF_VAR methods are allowed')
                except NameError:
                    print('An exception flew by!')
                    raise

            try:
                mse = get_mse(pred_matrix, validation_matrix)
            except ValueError:
                print('%d parameter combination is very bad.\nThe MSE is set to be default' % j)
                continue

            print('\t\tMSE=', mse)
            MSE[j, i-1] = mse

    MSE_result = MSE.sum(axis=1)
    return MSE_result



if __name__ == '__main__':

    time1 = time.time()
    real_cyc = pd.read_csv(r'D:\thesis\data\real_cyc.csv', index_col='starttime')
    array_real_cyc = np.array(real_cyc)

    n_groups = 5
    n_try_params = 30
    MSE = np.ones(shape=(n_try_params, n_groups-1))*100


    # specify parameters and distributions to sample from
    #
    # parameters for MF and PMF
    # param_dist = {'temp_bias_reg': uniform(),
    #               'geo_bias_reg': uniform(),
    #               'temp_reg': uniform(),
    #               'geo_reg': uniform(),
    #               }
    #
    # param_list = ParameterSampler(param_distributions=param_dist, n_iter=n_try_params, random_state=666)

    # parameters for PMF_VAR
    param_dist = {'temp_bias_reg': uniform(),
                  'geo_bias_reg': uniform(),
                  'temp_reg': uniform(),
                  'geo_reg': uniform(),
                  'var_reg': uniform()
                  }

    # parameters for Contingency table
    param_dist = {'temp_bias_reg': uniform(),
                  'geo_bias_reg': uniform(),
                  }


    param_list = ParameterSampler(param_distributions=param_dist, n_iter=n_try_params, random_state=666)

    # split the data set into 5 pieces, #test=1, # val=1, #train=3
    MSE_param = cross_val(data=array_real_cyc, cv=n_groups, n_iter=100, learning_rate=0.0003,
                          param_list=list(param_list), method='PMF_VAR')
    params = best_param(MSE_param, param_list=list(param_list), verbose=True)

    print('spend:', time.time()-time1, ' (s)')


