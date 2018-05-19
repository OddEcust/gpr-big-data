import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
import time

#read in airplane data which has been transformed previously when working with the same data set in R
df_airplane = pd.read_csv("../airline_standardized.csv")

print(df_airplane.describe())

# create sample size iterator
sample_set = np.arange(start=100, stop = len(df_airplane), step = 1000)

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
kernels = [k1,k2,k3,k4,k5,k6,k7,k8,k9]

# start the cycle
for kernel in kernels:
    name = "skilearn_airline_"+str(kernel)+".txt"
    with open(name, "a") as myfile:
        ## create a new file
        header = ["sample_size", "RMSE", "MAE","time_took", "kernel"]
        header = ",".join(header)
        myfile.write(header + "\n")

        for sample in sample_set:
            # sample airplane data set and divide it into train/test sets
            df_airplane_sample = df_airplane.sample(n = sample, replace = False)
            X = df_airplane_sample.drop('ArrDelay', axis=1)
            y = df_airplane_sample[["ArrDelay"]].copy()
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