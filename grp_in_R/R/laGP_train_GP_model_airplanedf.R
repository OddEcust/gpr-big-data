# Load all needed libraries ----------------------------------------------------
library(dplyr)
library(reshape2)
library(kernlab)
library(dummies)
library(laGP)

# read in airplane data set ----------------------------------------------------
df <- read.csv("../data_sources/Airline_data_set/airline_standardized.csv",
               stringsAsFactors = FALSE)

# MAE and RMSE  ---------------------------------------------------------------- 
source("R/calculate_MAE.R")
source("R/calculate_RMSE.R")

# Split data into train and test sets ------------------------------------------

## 80% of the sample size
smp_size <- floor(0.80 * nrow(df))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

df_train <- df[train_ind, ]
df_test  <- df[-train_ind, ]

# Main part: test various methods ----=-----------------------------------------

# create sample set size sequence from original data frame == df_train, starting
# with n=100 and increasing size by 10^4 or 10^6 until reaches nrow of data 
# frame
sample_size_first_part = seq(10^2, 10^6, by = 10^4)
sample_size_second_part = seq(10^6, nrow(df_train), by = 10^6)
sample_size = c(sample_size_first_part, sample_size_second_part, nrow(df_train))

# initialize empty data frame to record metrics
experiment_data_sample <- data.frame(sample_size = NA, 
                                     time.taken = NA,
                                     RMSE = NA,
                                     MAE = NA)
# create vector of methods:
methods <- c("alcray", "alc", "nn", "mspe")

# go through all sample sizes while it runs out of memory obtaining evaluation 
# metrics MAE and RMSE.

## loop will take a lot of time while going through all methods and sample sets
for (method in methods) {
  for (i in sample_size){
    # create sample of df_train
    sample_rows <- sample(nrow(df_train), size = i, replace = FALSE)
    x_sample <- df_train[sample_rows,(!(names(df_train) %in% "ArrDelay"))]
    y_sample <- df_train[sample_rows,"ArrDelay"]
    
    # crrate test sample of df_test because of transductive learning procedure
    if(i<nrow(df_test)){
      sample_rows_sample <- sample(nrow(df_test), size = i, replace = FALSE)
    } else{
      sample_rows_sample <- sample(nrow(df_test), size = nrow(df_test), 
                                   replace = FALSE)
    }
    x_test <- df_test[sample_rows_sample,(!(names(df_test) %in% "ArrDelay"))]
    
    # creating prior
    d_prior <- laGP::darg(list(mle=TRUE), data.frame(x_sample), 
                          samp.size=1000)
    
    #*start model training -----
    start.time <- Sys.time()
    tryCatch(
      gausspr_model <- laGP::aGP(X = x_sample, 
                                 Z = y_sample, 
                                 XX = x_test,
                                 d = d_prior,
                                 method = method,
                                 omp.threads=16),
      error = function(error){
        stop((paste("Error while creating model for sample with size of",
                    as.character(i)))
        )
      })
    #*end of model training ----
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "secs")
    
    # getting predictions and evaluation metrics -----
    predicted_values <- gausspr_model$mean
    
    RMSE_value = calculate_RMSE(df_test[sample_rows_sample,"ArrDelay"], 
                                predicted_values)
    MAE_value = calculate_MAE(df_test[sample_rows_sample,"ArrDelay"], 
                              predicted_values)
    
    experiment_data_sample <- rbind(experiment_data_sample,
                                    data.frame(sample_size = i,
                                               time.taken = time.taken,
                                               RMSE = RMSE_value,
                                               MAE = MAE_value))
    print(paste("sample data set with size of",as.character(i),", RSME:",
                as.character(RMSE_value),
                "in time", as.character(time.taken)))
  }
  file_name <- paste0("laGP_airline_", method,"_sample.csv")
  write.csv(experiment_data_sample, file_name, row.names = FALSE)
}
