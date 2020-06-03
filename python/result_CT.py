from loss import *
from CV_params import *
import pandas as pd
import time
from joblib import dump


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

# set parameters
num_iter = 150
learning = 0.0003


# Contingency Table without regularization
time1 = time.time()
CT = Contingency_table(train_matrix, test_matrix, learning='sgd',
                            temp_reg=0, geo_reg=0,
                            process=True, verbose=True)
CT.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix_CT = CT.predict_all()
mse_test = get_mse(pred_matrix_CT, test_matrix)
mse_train = get_mse(pred_matrix_CT, train_matrix)
CT.plot_learning_curve(title_add='Without Regularization',
                        MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train, mse_test))

print('spend: ', time.time()-time1, '(s)')


# Contingency Table with regularization
time2 = time.time()
CT2 = Contingency_table(train_matrix, test_matrix, learning='sgd',
                            temp_reg=0.0724, geo_reg=0.0302,
                            process=True, verbose=True)
CT2.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix_CT2 = CT2.predict_all()
mse_test2 = get_mse(pred_matrix_CT2, test_matrix)
mse_train2 = get_mse(pred_matrix_CT2, train_matrix)
CT2.plot_learning_curve(title_add='With Regularization',
                       MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train2, mse_test2))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix_CT, train_matrix, title_add='Contingency Table Without Regularization (train residuals)',
               df=n_count-167-1312+1, family='poisson')
model_diagnosis(pred_matrix_CT2, train_matrix, title_add='Contingency Table With Regularization (train residuals)',
               df=n_count-167-1312+1, family='poisson')

# test residuals plot
test_pearson_residuals(pred_matrix_CT, test_matrix, title_add='Contingency Table Without Regularization (matrix forecast)')
test_pearson_residuals(pred_matrix_CT2, test_matrix, title_add='Contingency Table With Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix_CT, train_observe_cyc, title_add='Contingency Table Without Regularization')
plot_inference_delta(pred_matrix_CT2, train_observe_cyc, title_add='Contingency Table With Regularization')

# save the model
dump(CT, r'D:\thesis\models\CT.joblib')
dump(CT2, r'D:\thesis\models\CT2.joblib')

print('spend: ', time.time()-time2, '(s)')
