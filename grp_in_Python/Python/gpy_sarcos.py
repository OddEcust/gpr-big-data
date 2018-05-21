import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
import time
import GPy

# Get the dataset here: http://www.gaussianprocess.org/gpml/data/
import scipy.io

# Load training set
train = scipy.io.loadmat("sarcos_inv.mat")
# Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations) + 1 Output
df_train = pd.DataFrame(train["sarcos_inv"][:, :22])

#column names 
feature_count = np.arange(start=1, stop = (22), step =1)
column_names = ["feature_" + str(s) for s in feature_count]
column_names.append('target_feature')
df_train.columns = column_names

# Load test set
test = scipy.io.loadmat("sarcos_inv_test.mat")
df_test = pd.DataFrame(test["sarcos_inv_test"][:, :22])
df_test.columns = column_names

# create sample size iterator
sample_set = np.arange(start=100, stop = len(df_train), step = 1000)

# start the cycle
#kernel_name = "RBF"
#kernel_name = "MLP"
kernel_name = "Matern52"

name = "gpy_airplane_"+str(kernel_name)+"_new.txt"
with open(name, "a") as myfile:
    ## create a new file
    header = ["sample_size", "RMSE","MAE","time_took", "kernel"]
    header = ",".join(header)
    myfile.write(header + "\n")

    for sample in sample_set:
        # sample airplane data set and divide it into train/test sets
        df_train_sarcos_sample = df_train.sample(n = sample, replace = False)
        X_train = df_train_sarcos_sample.drop('target_feature', axis=1)
        y_train = df_train_sarcos_sample[["target_feature"]].copy()
        
        kernel = GPy.kern.RBF(X_train.shape[1])
        #kernel = GPy.kern.MLP(X_train.shape[1], 1)
        #kernel = GPy.kern.Matern52(X_train.shape[1],1)
        
        start_time = time.clock()
        try:
            gp = GPy.models.GPRegression(X_train, y_train, kernel)
            gp.optimize(max_iters = 100)
        except:
            print("memory issue")
            break
        else:
            y_predict = gp.predict(df_test.drop('target_feature', axis=1).values)[0]
            end_time = time.clock()

            #from pandas series we are taking first value as RMSE always will be calculcated as one value
            y_test = df_test[["target_feature"]].copy()
            rmse_out =  np.sqrt(np.mean(np.power((y_test-np.array(y_predict)),2))).values[0]
            mae_out = np.mean(np.abs(y_test-np.array(y_predict))).values[0]
            time_took = end_time - start_time

            out_list = [str(len(X_train)), str(rmse_out), str(mae_out), str(time_took), str(kernel_name)]
            out_list = ",".join(out_list)
            myfile.write(out_list+ "\n")
            print(out_list)