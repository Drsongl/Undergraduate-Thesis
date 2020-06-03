from loss import *
from CV_params import *
import pandas as pd
import time
from joblib import dump


# read data
real_cyc = pd.read_csv(r'D:\thesis\data\real.csv', index_col='starttime')
observe_cyc = pd.read_csv(r'D:\thesis\data\observed.csv', index_col='starttime')
array_real_cyc = np.array(real_cyc)
array_observe_cyc = np.array(observe_cyc)

train_real_cyc = array_real_cyc[:-24, :]
test_real_cyc = array_real_cyc[-24:, :]
train_observe_cyc = array_observe_cyc[:-24, :]
test_observe_cyc = array_observe_cyc[-24:, :]

n_count = 167*1312 - np.sum(np.sum(real_cyc.isna()))

# split data
group = generate_sets(train_real_cyc, groups=5)
test_matrix, _, train_matrix = generate_matrix(data=train_real_cyc, groups=group, test_order=0)

# parameters
num_iter = 150
learning = 0.0001

# PMF_VAR
time4 = time.time()
'''
# without regularization
PMF_VAR = PoissonMF_VAR(train_matrix, test_matrix, n_factors=3, learning='sgd',
                            temp_reg=0, geo_reg=0, var_reg=0.7052,
                            temp_bias_reg=0, geo_bias_reg=0,
                            A1=1.04 * np.eye(3),
                            A2=-0.38 * np.eye(3),
                            process=True, verbose=True)
PMF_VAR.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix_PMF_VAR = PMF_VAR.predict_all()
mse_test = get_mse(pred_matrix_PMF_VAR, test_matrix)
mse_train = get_mse(pred_matrix_PMF_VAR, train_matrix)
PMF_VAR.plot_learning_curve(title_add='Without Regularization',
                            MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train, mse_test))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix_PMF_VAR, train_matrix, title_add='PMF+VAR Without Regularization (train residuals)',
               df=n_count-4*(167+1312)-1, family='poisson')

# test residuals plot
test_pearson_residuals(pred_matrix_PMF_VAR, test_matrix, title_add='PMF+VAR Without Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix_PMF_VAR, train_observe_cyc, title_add='PMF+VAR Without Regularization')

# save the model
dump(PMF_VAR, r'D:\thesis\models\PMF_VAR.joblib')
'''

# with regularization
PMF_VAR2 = PoissonMF_VAR(train_matrix, test_matrix, n_factors=3, learning='sgd',
                            temp_reg=0.1390, geo_reg=0.2117, var_reg=0.2118,
                            temp_bias_reg=0.0015, geo_bias_reg=0.0409,
                            A1=1.04 * np.eye(3),
                            A2=-0.38 * np.eye(3),
                            process=True, verbose=True)
PMF_VAR2.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix_PMF_VAR2 = PMF_VAR2.predict_all()
mse_test2 = get_mse(pred_matrix_PMF_VAR2, test_matrix)
mse_train2 = get_mse(pred_matrix_PMF_VAR2, train_matrix)

PMF_VAR2.plot_learning_curve(title_add='With Regularization',
                            MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train2, mse_test2))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix_PMF_VAR2, train_matrix, title_add='PMF+VAR With Regularization (train residuals)',
               df=n_count-4*(167+1312)-1, family='poisson')

# test residuals plot
test_pearson_residuals(pred_matrix_PMF_VAR2, test_matrix, title_add='PMF+VAR With Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix_PMF_VAR2, train_observe_cyc, title_add='PMF+VAR With Regularization')

# save the model
dump(PMF_VAR2, r'D:\thesis\models\PMF_VAR2.joblib')

print('spend: ', time.time()-time4, '(s)')
