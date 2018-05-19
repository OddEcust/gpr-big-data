import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
import time

# Load data
df_abalone = pd.read_csv("/home/ubuntu/abalone_data_set.csv")
# create dummy variables for sex column
df_abalone_transformed = pd.get_dummies(df_abalone)

print(df_abalone.head())

# create sample size iterator
sample_set = np.arange(start=100, stop = df_abalone_transformed.shape[0], step = 300)

# start the cycle
for kernel in kernels:
    name = "skilearn_abalone_"+str(kernel)+".txt"
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

            start_time = time.clock()
            try:
                gp = gaussian_process.GaussianProcessRegressor(kernel = kernel)
                gp.fit(X_train, y_train)
            except:
                print("memory issue")
                break
            else:
                y_pred = gp.predict(X_test, return_std=False)
                end_time = time.clock()

                #from pandas series we are taking first value as RMSE always will be calculcated as one value
                rmse_out =  np.sqrt(np.mean(np.power((y_test-y_pred),2))).values[0]
                mae_out = np.mean(np.abs(y_test-y_pred)).values[0]
                time_took = end_time - start_time

                out_list = [str(len(X_train)), str(rmse_out), str(mae_out), str(time_took), str(kernel)]
                out_list = ",".join(out_list)
                myfile.write(out_list+ "\n")
                print(out_list)