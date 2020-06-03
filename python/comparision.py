from loss import *
from CV_params import *
import pandas as pd
import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import ParameterSampler
from random import shuffle

with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0



real_cyc = pd.read_csv(r'D:\thesis\data\real_cyc.csv', index_col='starttime')
array_real_cyc = np.array(real_cyc)


group = generate_sets(array_real_cyc, groups=5)
test_matrix, _, train_matrix = generate_matrix(data=array_real_cyc, groups=group, test_order=0)
'''
# MF
time1 = time.time()
MF = ExplicitMF(train_matrix, test_matrix, n_factors=3, learning='sgd',
                    item_fact_reg=0.0125, user_fact_reg=0.0808,
                    item_bias_reg=1.9189, user_bias_reg=0.0729,
                    process=True, verbose=True)
MF.fit(n_iter=150, learning_rate=0.001)
pred_matrix_MF = MF.predict_all()
MSE_MF = get_mse(pred_matrix_MF, test_matrix)
MF.plot_learning_curve(mse=MSE_MF)

print('spend: ', time.time()-time1, '(s)')
print('MSE = ', MSE_MF)

# PMF
time2 = time.time()
PMF = PoissonMF(train_matrix, test_matrix, n_factors=3, learning='sgd',
                    temp_reg=0.2350, geo_reg=0.4308,
                    temp_bias_reg=0.0165, geo_bias_reg=0.2566,
                    process=True, verbose=True)
PMF.fit(n_iter=150, learning_rate=0.0005)
pred_matrix_PMF = PMF.predict_all()
MSE_PMF = get_mse(pred_matrix_PMF, test_matrix)
PMF.plot_learning_curve(mse=MSE_PMF)

print('spend: ', time.time()-time2, '(s)')
print('MSE = ', MSE_PMF)
'''

# PMF_VAR, parameters 1
time3 = time.time()
PMF_VAR = PoissonMF_VAR(train_matrix, test_matrix, n_factors=3, learning='sgd',
                            temp_reg=0.1, geo_reg=0.4136, var_reg=0.5081,
                            temp_bias_reg=0.0488, geo_bias_reg=0.0127,
                            process=True, verbose=True)
PMF_VAR.fit(n_iter=150, learning_rate=0.0003)
pred_matrix_PMF_VAR = PMF_VAR.predict_all()
MSE_VAR = get_mse(pred_matrix_PMF_VAR, test_matrix)
PMF_VAR.plot_learning_curve(mse=MSE_VAR)

print('spend: ', time.time()-time3, '(s)')
print('MSE = ', MSE_VAR)


# PMF_VAR, parameters 2
time4 = time.time()
PMF_VAR2 = PoissonMF_VAR(train_matrix, test_matrix, n_factors=3, learning='sgd',
                            temp_reg=0.1, geo_reg=0.4136, var_reg=0.5081,
                            temp_bias_reg=0.0488, geo_bias_reg=0.0127,
                            process=True, verbose=True)
PMF_VAR2.fit(n_iter=150, learning_rate=0.0003)
pred_matrix_PMF_VAR2 = PMF_VAR2.predict_all()
MSE_VAR2 = get_mse(pred_matrix_PMF_VAR2, test_matrix)
PMF_VAR2.plot_learning_curve(mse=MSE_VAR2)

print('spend: ', time.time()-time3, '(s)')
print('MSE = ', MSE_VAR2)
