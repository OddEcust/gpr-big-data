library(readr)
library(reshape2)
library(ggplot2)
library(scales) # to access break formatting functions

## read in all data sets and align column names ----
sci_abalone_evaluation <- 
  read_csv(paste0(file.path("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                            "grp_in_Python/output/skilearn_abalone_output.csv"))
           )
names(sci_abalone_evaluation)

sci_SARCOS_evaluation <- 
  read_csv(paste0(file.path("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                            "grp_in_Python/output/skilearn_SARCOS_output.csv"))
           )
names(sci_SARCOS_evaluation)

sci_airplane_evaluation <- 
  read_csv(paste0(file.path("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                            "grp_in_Python/output/skilearn_airline_output.csv")
                  )
           )
names(sci_airplane_evaluation)

# create summary data frame -----
df <- rbind(sci_abalone_evaluation, sci_SARCOS_evaluation,
            sci_airplane_evaluation) %>%
  select(data, kernel,sample_size,time_took,RMSE,MAE) %>%
  group_by(data, kernel,sample_size) %>%
  summarize(time_took = median(time_took, na.rm = TRUE)/60,
            RMSE = median(RMSE, na.rm = TRUE),
            MAE = median(MAE, na.rm = TRUE)
  ) %>%
  ungroup()

# keep only complete cases ---- 
df <- df[complete.cases(df),]

# write summary data frame for future use ----
write.csv(df, 
          file.path(paste0("C:/cygwin64/home/agrita.garnizone/gpr-big-data/",
                          "grp_in_Python/output/skilearn_output_summary.csv")),
          row.names = FALSE)

# create vizualizations | data ---- 
df_melted <- reshape2::melt(df, id.vars = c("data", "kernel", "sample_size"))

df_melted_abalone <- df_melted[which(df_melted$data == "abalone"),]
df_melted_sarcos <- df_melted[which(df_melted$data == "sarcos"),]
df_melted_airplane <- df_melted[which(df_melted$data == "airplane"),]

# create vizualizations ABALONE | ggplot ----
df_input <- df_melted_abalone %>%
  select(kernel, sample_size, variable, value) %>%
  filter(variable %in% c("RMSE", "MAE"))

ggplot(data = df_input, aes(x=sample_size, y = value, color = kernel)) +
  geom_line()+ geom_point() +
  facet_grid(variable~., scales = "free", 
             labeller = 
               labeller(variable =c("MAE" = "MAE",
                                    "RMSE" = "RMSE"))) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  ylab("Vērtība")+
  xlab("Izlases apjoms") +
  theme_bw()+
  theme(axis.title=element_text(size=14),
        legend.title = element_blank(),
        legend.text = element_text(size=12))
        legend.text = element_blank())


df_input <- df_melted_abalone %>%
  select(kernel, sample_size, variable, value) %>%
  filter(variable %in% c("time_took"))

ggplot(data = df_input, aes(x=sample_size, y = value, color = kernel)) +
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
        legend.text = element_blank())


# create vizualizations AIRPLANE | ggplot ----
df_input <- df_melted_airplane %>%
  select(kernel, sample_size, variable, value) %>%
  filter(variable %in% c("RMSE", "MAE"))

ggplot(data = df_input, aes(x=sample_size, y = value, color = kernel)) +
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
        legend.text = element_blank())
        #legend.position = "bottom")


df_input <- df_melted_airplane %>%
  select(kernel, sample_size, variable, value) %>%
  filter(variable %in% c("time_took"))

ggplot(data = df_input, aes(x=sample_size, y = value, color = kernel)) +
  geom_line()+ geom_point() +
  facet_grid(variable~., scales = "free", 
             labeller = 
               labeller(variable =c("time_took" = "Apmācīšanās laiks"))) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  ylab("Laiks (minūtes)")+
  xlab("Izlases apjoms") +
  theme_bw()+
  theme(axis.title=element_text(size=14),
        legend.title = element_blank(),
        legend.text = element_blank())

# create vizualizations SARCOS | ggplot ----
df_input <- df_melted_sarcos %>%
  select(kernel, sample_size, variable, value) %>%
  filter(variable %in% c("RMSE", "MAE"))

ggplot(data = df_input, aes(x=sample_size, y = value, color = kernel)) +
  geom_line()+ geom_point() +
  facet_grid(variable~., scales = "free", 
             labeller = 
               labeller(variable =c("MAE" = "MAE",
                                    "RMSE" = "RMSE"))) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  ylab("Vērtība")+
  xlab("Izlases apjoms") +
  theme_bw()+
  theme(axis.title=element_text(size=14),
        legend.title = element_blank(),
        legend.text = element_blank())


df_input <- df_melted_sarcos %>%
  select(kernel, sample_size, variable, value) %>%
  filter(variable %in% c("time_took"))

ggplot(data = df_input, aes(x=sample_size, y = value, color = kernel)) +
  geom_line()+ geom_point() +
  facet_grid(variable~., scales = "free", 
             labeller = 
               labeller(variable =c("time_took" = "Apmācīšanās laiks"))) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  ylab("Laiks (minūtes)")+
  xlab("Izlases apjoms") +
  theme_bw()+
  theme(axis.title=element_text(size=14),
        legend.title = element_blank(),
        legend.text = element_blank())
