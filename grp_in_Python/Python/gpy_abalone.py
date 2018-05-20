import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
import time
import GPy

##ABALONE
df_abalone = pd.read_csv("../abalone_data_set.csv")
df_abalone_transformed = pd.get_dummies(df_abalone)

# start the cycle
#kernel = "RBF"
#kernel_name = "MLP"
kernel_name = "Matern52"

name = "gpy_abalone_"+str(kernel_name)+".txt"
with open(name, "a") as myfile:
    ## create a new file
    header = ["sample_size", "RMSE","MAE","time_took", "kernel"]
    header = ",".join(header)
    myfile.write(header + "\n")

    for sample in sample_set:
        # sample airplane data set and divide it into train/test sets
        df_abalone_sample = df_abalone_transformed.sample(n = sample, replace = False)
        X = df_abalone_sample.drop('rings', axis=1)
        y = df_abalone_sample[["rings"]].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #kernel = GPy.kern.RBF(X_train.shape[1], ARD=1)
        #kernel = GPy.kern.MLP(X_train.shape[1], 1)
        kernel = GPy.kern.Matern52(X_train.shape[1],1)
        
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