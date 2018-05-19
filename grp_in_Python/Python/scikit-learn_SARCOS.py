import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
import time
import scipy.io #for .mat files

# Load training set
train = scipy.io.loadmat("../sarcos_inv.mat")
# Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations) + 1 tagert value
df_train = pd.DataFrame(train["sarcos_inv"][:, :22])

# Change column names 
feature_count = np.arange(start=1, stop = (22), step =1)
column_names = ["feature_" + str(s) for s in feature_count]
column_names.append('target_feature')

# Assign column names 
df_train.columns = column_names

# Load test set
test = scipy.io.loadmat("sarcos_inv_test.mat")
df_test = pd.DataFrame(test["sarcos_inv_test"][:, :22])
df_test.columns = column_names

# create sample size iterator
sample_set = np.arange(start=100, stop = len(df_train), step = 1000)

# create kernel iterator
k1 = RationalQuadratic(alpha = 0.1, length_scale = 1)
k2 = RationalQuadratic(alpha = 0.1, length_scale = 0.5)
k3 = RationalQuadratic(alpha = 2, length_scale = 0.5)
k4 = ExpSineSquared(length_scale = 0.1, periodicity = 5)
k5 = Matern(length_scale=1.1, nu=3 / 2)
k6 = Matern(length_scale=1.1, nu=5 / 2)
k7 = RBF(length_scale = 1)
k8 = RBF(length_scale = 0.1)
k9 = RBF(length_scale = 0.1)*RationalQuadratic(alpha = 0.1, length_scale = 0.5)
kernels = [k7,k8,k9,k1,k2,k3,k4,k5,k6]

# start the cycle
for kernel in kernels:
    name = "skilearn_SARCOS_"+str(kernel)+".txt"
    with open(name, "a") as myfile:
        ## create a new file
        header = ["sample_size", "RMSE", "MAE","time_took", "kernel"]
        header = ",".join(header)
        myfile.write(header + "\n")

        for sample in sample_set:
            # sample airplane data set and divide it into train/test sets
            df_train_sarcos_sample = df_train.sample(n = sample, replace = False)
            X_train = df_train_sarcos_sample.drop('target_feature', axis=1)
            y_train = df_train_sarcos_sample[["target_feature"]].copy()

            start_time = time.clock()
            try:
                gp = gaussian_process.GaussianProcessRegressor(kernel = kernel)
                gp.fit(X_train, y_train)
            except:
                print("memory issue")
                break
            else:
                y_pred = gp.predict(df_test.drop('target_feature', axis=1), return_std=False)
                end_time = time.clock()

                #from pandas series we are taking first value as RMSE always will be calculcated as one value
                y_test = df_test[["target_feature"]].copy()
                rmse_out =  np.sqrt(np.mean(np.power((y_test-y_pred),2))).values[0]
                mae_out = np.mean(np.abs(y_test-y_pred)).values[0]
                time_took = end_time - start_time

                out_list = [str(len(X_train)), str(rmse_out), str(mae_out), str(time_took), str(kernel)]
                out_list = ",".join(out_list)
                myfile.write(out_list+ "\n")
                print(out_list)