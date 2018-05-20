library(readr)
library(reshape2)
library(ggplot2)
library(scales) # to access break formatting functions

## read in all data sets and align column names ----
kernlab_abalone_evaluation <- 
  read_csv(paste0(file.path("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                            "grp_in_R/output/kernlab_abalone_evaluation.csv")))
kernlab_abalone_evaluation$data<- "abalone"
names(kernlab_abalone_evaluation)

kernlab_SARCOS_evaluation <- 
  read_csv(paste0(file.path("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                            "grp_in_R/output/kernlab_sarcos_evaluation.csv")))
kernlab_SARCOS_evaluation$data <- "SARCOS"
names(kernlab_SARCOS_evaluation)

kernlab_airplane_evaluation <- 
  read_csv(paste0(file.path("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                            "grp_in_R/output/kernlab_airplane_evaluation.csv")))
kernlab_airplane_evaluation$data <- "airplane"
kernlab_airplane_evaluation$time.taken.test <- NA
names(kernlab_airplane_evaluation)

# create summary data frame -----
df <- rbind(kernlab_abalone_evaluation, kernlab_SARCOS_evaluation,
            kernlab_airplane_evaluation) %>%
  select(data, method,sample_size,time.taken,RMSE,MAE) %>%
  group_by(data, method,sample_size) %>%
  summarize(time.taken = median(time.taken, na.rm = TRUE),
         RMSE = median(RMSE, na.rm = TRUE),
         MAE = median(MAE, na.rm = TRUE)
         ) %>%
  ungroup()

# keep only complete cases ---- 
df <- df[complete.cases(df),]

table(df$data,df$method)

# write summary data frame for future use ----
write.csv(df, file.path(paste0("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                               "grp_in_R/output/kernlab_output_summary.csv")),
          row.names = FALSE)
                        
# create vizualizations | data ---- 
df_melted <- reshape2::melt(df, id.vars = c("data", "method", "sample_size"))

df_melted_abalone <- df_melted[which(df_melted$data == "abalone"),]
df_melted_sarcos <- df_melted[which(df_melted$data == "SARCOS"),]
df_melted_airplane <- df_melted[which(df_melted$data == "airplane"),]

# create vizualizations ABALONE | ggplot ----
df_input <- df_melted_abalone %>%
    select(method, sample_size, variable, value) %>%
    filter(variable %in% c("RMSE", "MAE"))
  
  if(!is.na(method_seperate)){
   df_input =
     df_input %>%
     mutate(variable = ifelse(method == method_seperate,
                              paste(variable, method_seperate, sep="_"),
                               as.character(variable)))
  }
  
  ggplot(data = df_input, aes(x=sample_size, y = value, color = method)) +
  geom_line()+ geom_point() +
  facet_grid(variable~., scales = "free", 
             labeller = 
               labeller(variable =c("MAE" = "MAE",
                                    "MAE_anovadot" = "MAE, anovadot",
                                    "RMSE" = "RMSE",
                                    "RMSE_anovadot" = "RMSE, anovadot"))) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  ylab("Vērtība")+
  xlab("Izlases apjoms") +
  theme_bw()+
  theme(axis.title=element_text(size=14),
        legend.title = element_blank(),
        legend.position = "bottom")


df_input <- df_melted_abalone %>%
    select(method, sample_size, variable, value) %>%
    filter(variable %in% c("time.taken"))

  ggplot(data = df_input, aes(x=sample_size, y = value/60, color = method)) +
    geom_line(size = .8, alpha = .9)+ geom_point() +
    facet_grid(variable~., scales = "free", 
               labeller = 
                 labeller(variable =c("time.taken" = "Apmācīšanās laiks"))) +
    scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x)))+
    ylab("Laiks (minūtes)")+
    xlab("Izlases apjoms") +
    theme_bw()+
    theme(axis.title=element_text(size=14),
          legend.title = element_blank(),
          legend.position = "bottom")

  
# create vizualizations AIRPLANE | ggplot ----
  df_input <- df_melted_airplane %>%
    select(method, sample_size, variable, value) %>%
    filter(variable %in% c("RMSE", "MAE"))
  
  ggplot(data = df_input, aes(x=sample_size, y = value, color = method,
                              size = method)) +
    geom_line()+ geom_point() +
    scale_size_manual(values = c(3, 1.5, 1))+
    facet_grid(variable~., scales = "free", 
               labeller = 
                 labeller(variable =c("MAE" = "MAE",
                                      "MAE_anovadot" = "MAE, anovadot",
                                      "RMSE" = "RMSE",
                                      "RMSE_anovadot" = "RMSE, anovadot"))) +
    scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x)))+
    ylab("Vērtība")+
    xlab("Izlases apjoms") +
    theme_bw()+
    theme(axis.title=element_text(size=14),
          legend.title = element_blank(),
          legend.position = "bottom")
  
  
  df_input <- df_melted_airplane %>%
    select(method, sample_size, variable, value) %>%
    filter(variable %in% c("time.taken"))
  
  ggplot(data = df_input, aes(x=sample_size, y = value/60, color = method,
                              size = method)) +
    geom_line()+ geom_point() +
    scale_size_manual(values = c(3, 1.5, 1))+
    facet_grid(variable~., scales = "free", 
               labeller = 
                 labeller(variable =c("time.taken" = "Apmācīšanās laiks"))) +
    scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x)))+
    ylab("Laiks (minūtes)")+
    xlab("Izlases apjoms") +
    theme_bw()+
    theme(axis.title=element_text(size=14),
          legend.title = element_blank(),
          legend.position = "bottom")
  
  # create vizualizations SARCOS | ggplot ----
  df_input <- df_melted_sarcos %>%
    select(method, sample_size, variable, value) %>%
    filter(variable %in% c("RMSE", "MAE"))
  
  ggplot(data = df_input, aes(x=sample_size, y = value, color = method,
                              size = method)) +
    geom_line()+ geom_point() +
    scale_size_manual(values = c(2, 1, 1))+
    facet_grid(variable~., scales = "free", 
               labeller = 
                 labeller(variable =c("MAE" = "MAE",
                                      "MAE_anovadot" = "MAE, anovadot",
                                      "RMSE" = "RMSE",
                                      "RMSE_anovadot" = "RMSE, anovadot"))) +
    scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x)))+
    ylab("Vērtība")+
    xlab("Izlases apjoms") +
    theme_bw()+
    theme(axis.title=element_text(size=14),
          legend.title = element_blank(),
          legend.position = "bottom")
  
  
  df_input <- df_melted_sarcos %>%
    select(method, sample_size, variable, value) %>%
    filter(variable %in% c("time.taken"))
  
  ggplot(data = df_input, aes(x=sample_size, y = value/60, color = method,
                              size = method)) +
    geom_line()+ geom_point() +
    scale_size_manual(values = c(1.5 ,2, 1))+
    facet_grid(variable~., scales = "free", 
               labeller = 
                 labeller(variable =c("time.taken" = "Apmācīšanās laiks"))) +
    scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x)))+
    ylab("Laiks (minūtes)")+
    xlab("Izlases apjoms") +
    theme_bw()+
    theme(axis.title=element_text(size=14),
          legend.title = element_blank(),
          legend.position = "bottom")
  