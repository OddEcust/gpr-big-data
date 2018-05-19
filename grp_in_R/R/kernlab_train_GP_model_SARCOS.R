# Load all needed libraries ----------------------------------------------------
library(dplyr)
library(reshape2)
library(kernlab)

# read in SARCOS data set ----------------__------------------------------------
# data set for training
df_sarcos <- 
 rmatio::read.mat("http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat") %>%
 as.data.frame()
df_sarcos <- df_sarcos[,1:22]
names(df_sarcos) <- c(paste("feature", 1:21, sep = "_"), "target")

# data set for testing
df_sarcos_test <- 
 rmatio::read.mat("http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat") %>%
 as.data.frame()
df_sarcos_test <- df_sarcos_test[,1:22]
names(df_sarcos_test) <- c(paste("feature", 1:21, sep = "_"), "target")

# MAE and RMSE  ---------------------------------------------------------------- 
source("R/calculate_MAE.R")
source("R/calculate_RMSE.R")

# Main part: test various kernels ----=-----------------------------------------

# create sample set size sequence from original data frame == df_sarcos, 
#starting with n=100 and increasing size by 500 until reaches nrow of data frame
sample_size = c(seq(100, nrow(df_sarcos), by = 500), df_sarcos)

# initialize empty data frame to record metrics
experiment_data <- data.frame(sample_size = NA, 
                              time.taken = NA,
                              RMSE = NA,
                              MAE = NA,
                              time.taken.test = NA)

# create vector of kernels:
kernels <- c("rbfdot", "laplacedot", "vanilladot")


# go through all sample sizes while it runs out of memory
# loop will take some time while going through all sample sets
for (kernel in kernels) {
  for (i in sample_size){
    sample_rows <- sample(nrow(df_sarcos), size = i, replace = FALSE)
    x_sample <- df_sarcos[sample_rows,(!(names(df_sarcos) %in% "target"))]
    y_sample <- df_sarcos[sample_rows,"target"]
    
    start.time <- Sys.time()
    tryCatch(
      gausspr_model <- gausspr(x_sample, y_sample, variance.model=T,
                               kerne=kernel),
      error = function(error){
        print((paste("Error while creating model for sample with size of",
                     as.character(i))))
        break
      })
    
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "secs")
    
    start.time <- Sys.time()
    predicted_values = predict(gausspr_model, 
                               df_sarcos_test[,(!(names(df_sarcos_test) %in% 
                                                            "target"))]) %>%
                       as.vector()
    end.time <- Sys.time()
    time.taken.test <- difftime(end.time, start.time, units = "secs")
    
    RMSE_value = calculate_RMSE(df_sarcos_test[,"target"], predicted_values)
    MAE_value = calculate_MAE(df_sarcos_test[,"target"], predicted_values)
    
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
  
  file_name <- paste0("kernlab_SARCOS_", kernel,"_sample.csv")
  write.csv(experiment_data, kernel,
            row.names = FALSE)
}
