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

# Main part: test various kernels ----=-----------------------------------------

# create sample set size sequence from original data frame == df_train, starting
# with n=100 and increasing size by 10^4 or 10^6 until reaches nrow of data 
# frame
sample_size_first_part = seq(10^2, 10^6, by = 10^(3))
sample_size_second_part = seq(10^6, nrow(df_train), by = 10^6)
sample_size = c(sample_size_first_part, sample_size_second_part, nrow(df_train))

# initialize empty data frame to record metrics
experiment_data <- data.frame(sample_size = NA, 
                              time.taken = NA,
                              RMSE = NA,
                              MAE = NA,
                              time.taken.test = NA)

# create vector of kernels:
methods <- c("rbfdot", "laplacedot", "vanilladot")


# go through all sample sizes while it runs out of memory
# loop will take some time while going through all sample sets
for (kernel in kernels) {
  for (i in sample_size){
    sample_rows <- sample(nrow(df_train), size = i, replace = FALSE)
    x_sample <- df_train[sample_rows,(!(names(df_train) %in% "ArrDelay"))]
    y_sample <- df_train[sample_rows,"ArrDelay"]
    
    start.time <- Sys.time()
    tryCatch(
      gausspr_model <- gausspr(x_sample, y_sample, variance.model=T,
                               kerne='rbfdot',
                               kpar=list(sigma = 0.005)),
      error = function(error){
        print((paste("Error while creating model for sample with size of",
                     as.character(i))))
        break
      })
    
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "secs")
    
    start.time <- Sys.time()
    predicted_values = predict(gausspr_model, df_test[,(!(names(df_test) %in% 
                                                            "ArrDelay"))]) %>%
      as.vector()
    end.time <- Sys.time()
    time.taken.test <- difftime(end.time, start.time, units = "secs")
    
    RMSE_value = calculate_RMSE(df_test[,"ArrDelay"], predicted_values)
    MAE_value = calculate_MAE(df_test[,"ArrDelay"], predicted_values)
    
    experiment_data <- rbind(experiment_data,
                             data.frame(sample_size = i,
                                        time.taken = time.taken,
                                        RMSE = RMSE_value,
                                        MAE = MAE_value,
                                        time.taken.test = time.taken.test))
    print(paste("sample data set with size of",as.character(i),", RSME:",
                as.character(RMSE_value),
                "in time", as.character(time.taken)))
  }
  
  file_name <- paste0("kernlab_airline_", kernel,"_sample.csv")
  write.csv(experiment_data, kernel,
            row.names = FALSE)
}