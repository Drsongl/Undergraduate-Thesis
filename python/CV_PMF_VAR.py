from CV_params import *

time1 = time.time()
real_cyc = pd.read_csv(r'D:\thesis\data\real.csv', index_col='starttime')
array_real_cyc = np.array(real_cyc)

n_groups = 5
n_try_params = 30
num_iter = 25
learning = 0.0001
MSE = np.ones(shape=(n_try_params, n_groups-1))*100


# specify parameters and distributions to sample from
param_dist = {'temp_bias_reg': uniform(),
              'geo_bias_reg': uniform(),
              'temp_reg': uniform(),
              'geo_reg': uniform(),
              'var_reg': uniform()
              }


param_list = ParameterSampler(param_distributions=param_dist, n_iter=n_try_params, random_state=888)  # 666 or 888

# split the data set into 5 pieces, #test=1, # val=1, #train=3
MSE_param = cross_val(data=array_real_cyc, cv=n_groups, n_iter=num_iter, learning_rate=learning,
                      param_list=list(param_list), method='PMF_VAR')
params = best_param(MSE_param, param_list=list(param_list), verbose=True)

print('spend:', time.time()-time1, ' (s)')


