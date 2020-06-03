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

# MF without regularization
MF = ExplicitMF(train_matrix, test_matrix, n_factors=3, learning='sgd',
                    item_bias_reg=0, user_bias_reg=0,
                    item_fact_reg=0, user_fact_reg=0,
                    process=True, verbose=True)
MF.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix = MF.predict_all()
mse_test = get_mse(pred_matrix, test_matrix)
mse_train = get_mse(pred_matrix, train_matrix)
MF.plot_learning_curve(title_add='Without Regularization',
                           MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train, mse_test))



# MF with regularization
MF2 = ExplicitMF(train_matrix, test_matrix, n_factors=3, learning='sgd',
                    item_bias_reg=0.0051, user_bias_reg=0.1109,
                    item_fact_reg=0.1129, user_fact_reg=0.2477,
                    process=True, verbose=True)
MF2.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix2 = MF2.predict_all()
mse_test2 = get_mse(pred_matrix2, test_matrix)
mse_train2 = get_mse(pred_matrix2, train_matrix)
MF2.plot_learning_curve(title_add='With Regularization',
                            MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train2, mse_test2))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix, train_matrix, title_add='Matrix Factorization Without Regularization (train residuals)',
               df=n_count-4*(167+1312)-1, family='gaussian')
model_diagnosis(pred_matrix2, train_matrix, title_add='Matrix Factorization With Regularization (train residuals)',
               df=n_count-4*(167+1312)-1, family='gaussian')

# test residuals plot
test_pearson_residuals(pred_matrix, test_matrix, title_add='Matrix Factorization Without Regularization (matrix forecast)')
test_pearson_residuals(pred_matrix2, test_matrix, title_add='Matrix Factorization With Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix, train_observe_cyc, title_add='Matrix Factorization Without Regularization')
plot_inference_delta(pred_matrix2, train_observe_cyc, title_add='Matrix Factorization With Regularization')

# save the model
dump(MF, r'D:\thesis\models\MF.joblib')
dump(MF2, r'D:\thesis\models\MF2.joblib')

print('spend: ', time.time()-time2, '(s)')
