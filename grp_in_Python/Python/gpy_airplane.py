import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
import time
import GPy

#read in airplane data which has been transformed previously when working with the same data set in R
df_airplane = pd.read_csv("/home/ubuntu/airline_standardized.csv")

# create sample size iterator
sample_set = np.arange(start=100, stop = len(df_airplane), step = 1000)

# start the cycle
kernel_name = "RBF"
#kernel_name = "MLP"
#kernel_name = "Matern52"

name = "gpy_airplane_"+str(kernel_name)+".txt"
with open(name, "a") as myfile:
    ## create a new file
    header = ["sample_size", "RMSE","MAE","time_took", "kernel"]
    header = ",".join(header)
    myfile.write(header + "\n")

    for sample in sample_set:
        # sample airplane data set and divide it into train/test sets
        df_airplane_sample = df_airplane.sample(n = sample, replace = False)
        X = df_airplane_sample.drop('ArrDelay', axis=1)
        y = df_airplane_sample[["ArrDelay"]].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        kernel = GPy.kern.RBF(X_train.shape[1], ARD=1)
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
            y_predict = gp.predict(X_test.values)[0]
            end_time = time.clock()

            #from pandas series we are taking first value as RMSE always will be calculcated as one value
            rmse_out =  np.sqrt(np.mean(np.power((y_test-np.array(y_predict)),2))).values[0]
            mae_out = np.mean(np.abs(y_test-np.array(y_predict))).values[0]
            time_took = end_time - start_time

            out_list = [str(len(X_train)), str(rmse_out), str(mae_out), str(time_took), str(kernel_name)]
            out_list = ",".join(out_list)
            myfile.write(out_list+ "\n")
            print(out_list)