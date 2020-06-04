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

n_count = 167*1312 

group = generate_sets(train_real_cyc, groups=5)
test_matrix, _, train_matrix = generate_matrix(data=train_real_cyc, groups=group, test_order=0)

# parameters
num_iter = 150
learning = 0.001

# PMF
time3 = time.time()
# without regularization
PMF = PoissonMF(train_matrix, test_matrix, n_factors=3, learning='sgd',
                    temp_reg=0, geo_reg=0,
                    temp_bias_reg=0, geo_bias_reg=0,
                    process=True, verbose=True)
PMF.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix_PMF = PMF.predict_all()
mse_test = get_mse(pred_matrix_PMF, test_matrix)
mse_train = get_mse(pred_matrix_PMF, train_matrix)
PMF.plot_learning_curve(title_add='Without Regularization',
                        MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train, mse_test))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix_PMF, train_matrix, title_add='Poisson Matrix Factorization Without Regularization (train residuals)',
               df=n_count-4*(167+1312)-1, family='poisson')

# test residuals plot
test_pearson_residuals(pred_matrix_PMF, test_matrix, title_add='Poisson Matrix Factorization Without Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix_PMF, train_observe_cyc, title_add='Poisson Matrix Factorization Without Regularization')

# save the model
dump(PMF, r'D:\thesis\models\PMF.joblib')


# with regularization
PMF2 = PoissonMF(train_matrix, test_matrix, n_factors=3, learning='sgd',
                    temp_reg=0.2350, geo_reg=0.4308,
                    temp_bias_reg=0.0165, geo_bias_reg=0.2566,
                    process=True, verbose=True)
PMF2.fit(n_iter=num_iter, learning_rate=learning)
pred_matrix_PMF2 = PMF2.predict_all()
mse_test2 = get_mse(pred_matrix_PMF2, test_matrix)
mse_train2 = get_mse(pred_matrix_PMF2, train_matrix)
PMF2.plot_learning_curve(title_add='With Regularization',
                         MSE='\nEnd of Iteration, MSE(train)=%.3f MSE(test)=%.3f' % (mse_train2, mse_test2))

# model diagnosis, residuals plot
model_diagnosis(pred_matrix_PMF2, train_matrix, title_add='Poisson Matrix Factorization With Regularization (train residuals)',
               df=n_count-4*(167+1312)-1, family='poisson')

# test residuals plot
test_pearson_residuals(pred_matrix_PMF2, test_matrix, title_add='Poisson Matrix Factorization With Regularization (matrix forecast)')

# observed residuals plot
plot_inference_delta(pred_matrix_PMF2, train_observe_cyc, title_add='Poisson Matrix Factorization With Regularization')

# save the model
dump(PMF2, r'D:\thesis\models\PMF2.joblib')

print('spend: ', time.time()-time3, '(s)')
