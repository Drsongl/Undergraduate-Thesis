from loss import *
from CV_params import *
import pandas as pd
import time
from joblib import dump


time2 = time.time()
# read and process data
real_cyc = pd.read_csv(r'D:\thesis\data\real.csv', index_col='starttime')
observe_cyc = pd.read_csv(r'D:\thesis\data\observed.csv', index_col='starttime')
array_real_cyc = np.array(real_cyc)
array_observe_cyc = np.array(observe_cyc)

train_real_cyc = array_real_cyc[:-24, :]
test_real_cyc = array_real_cyc[-24:, :]
train_observe_cyc = array_observe_cyc[:-24, :]
test_observe_cyc = array_observe_cyc[-24:, :]

n_count = 167*1312 - np.sum(np.sum(real_cyc.isna()))

group = generate_sets(train_real_cyc, groups=5)
test_matrix, _, train_matrix = generate_matrix(data=train_real_cyc, groups=group, test_order=0)

# parameters
num_iter = 150
learning = 0.001
'''
# MF without regularization
MFvar = MF_VAR(train_matrix, test_matrix, n_factors=3, var_reg=0.8596, learning='sgd', process=True, verbose=True)
MFvar.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix = MFvar.predict_all()
mse_test = get_mse(pred_matrix, test_matrix)
mse_train = get_mse(pred_matrix, train_matrix)
MFvar.plot_learning_curve(title_add='Without Regularization',
                           MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train, mse_test))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix, train_matrix, title_add='MF VAR Without Regularization (train residuals)',
                df=n_count-4*(167+1312)-1, family='gaussian')

# test residuals plot
test_pearson_residuals(pred_matrix, test_matrix, title_add='MF VAR Without Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix, train_observe_cyc, title_add='MF VAR Without Regularization')

# save the model
dump(MFvar, r'D:\thesis\models\MFvar.joblib')
'''

# MF with regularization
MFvar2 = MF_VAR(train_matrix, test_matrix, n_factors=3, learning='sgd',
                geo_reg=0.9161, temp_reg=0.0593, var_reg=0.6722,
                geo_bias_reg=0.1235, temp_bias_reg=0.2195,
                process=True, verbose=True)
MFvar2.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix2 = MFvar2.predict_all()
mse_test2 = get_mse(pred_matrix2, test_matrix)
mse_train2 = get_mse(pred_matrix2, train_matrix)
MFvar2.plot_learning_curve(title_add='With Regularization',
                            MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train2, mse_test2))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix2, train_matrix, title_add='MF VAR With Regularization (train residuals)',
                df=n_count-4*(167+1312)-1, family='gaussian')

# test residuals plot
test_pearson_residuals(pred_matrix2, test_matrix, title_add='MF VAR With Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix2, train_observe_cyc, title_add='MF VAR With Regularization')

# save the model
dump(MFvar2, r'D:\thesis\models\MFvar2.joblib')

print('spend: ', time.time()-time2, '(s)')
