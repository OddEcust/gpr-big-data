Exploratory analysis of datasets
================
Agrita Garnizone
May 4, 2018

Purpose of this Notebook is to perform exploratory analysis for multiple data sets which are used later on when building Gaussian Process Regression both in R and Python. All data sets are selected as they have been used in literature before.

Before importing and describing all data sets, necessary libraries are imported.

``` r
#for data manipulation and trasformation:
library(dplyr) # version: 0.7.2

# for melting data frames in form ggplot2 requires
library(reshape2)

# for data visualization
library(ggplot2)
library(gridExtra)

# for reading MatLab files (.mat). 
library(rmatio)  #version 0.12.0
```

The SARCOS dataset
------------------

SARCOS is the data set used by [Rasmussen and Williams (2006)](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) and later on also on [E.Snelson (2007)](http://www.gatsby.ucl.ac.uk/~snelson/thesis.pdf). The Data relates to an inverse dynamic problem for SARCOS anthropomorphic robot arm - i.e., map from the 21 dimensional joint position, velocity, and acceleration space to the torque at a single joint.

As stated by Rasmussen and Williams, the inputs are linearly rescaled to have zero mean and unit variance on the training set. The outputs were centered so as to have zero mean on the training set.

And just like [Rasmussen and Williams (2006)](http://www.gaussianprocess.org/gpml/chapters/RW.pdf), [E.Snelson (2007)](http://www.gatsby.ucl.ac.uk/~snelson/thesis.pdf), [A.Banerjee et al (2008)](https://arxiv.org/pdf/1106.5779.pdf) and many others - task is to map 21 input variables to the first of the seven torques.

``` r
# data set for training
df_sarcos <- rmatio::read.mat("http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat") %>%
             as.data.frame()
df_sarcos <- df_sarcos[,1:22]
names(df_sarcos) <- c(paste("feature", 1:21, sep = "_"), "target")

# data set for testing
df_sarcos_test <- rmatio::read.mat("http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat") %>%
                  as.data.frame()
df_sarcos_test <- df_sarcos_test[,1:22]
names(df_sarcos_test) <- c(paste("feature", 1:21, sep = "_"), "target")
```

Data set contains more than 44k observations and 22 dimensions as stated before.

``` r
dim(df_sarcos)
```

    ## [1] 44484    22

Before model creation, exploratory data analysis are performed. Starting with data value summary visualized with boxplots for each joint group - position, velocity and acceleration space. Violin plots helps to understand distribution of data, and within violins you can see drawn quantiles (three horizontal lines - Q1, Q2, Q3), and mean value +/- standard deviation (red dot with red lines).

``` r
df_sacros_visualization <- reshape2::melt(df_sarcos[,1:21]) %>%
    dplyr::mutate(group = 
            ifelse(variable %in% c(paste("feature", 1:7, sep="_")),"position",
            ifelse(variable %in% c(paste("feature", 8:14, sep="_")),"velocity",
                   "acceleration")))
```

    ## No id variables; using all as measure variables

``` r
ggplot(data = 
   df_sacros_visualization[which(df_sacros_visualization$group == "position"),], 
        aes(x=variable, y=value)) + 
   geom_violin(aes(fill=variable), draw_quantiles = c(0.25, 0.5, 0.75)) +
   scale_fill_brewer(palette="Pastel1") +
   theme_bw()+
   theme(axis.text.x=element_text(angle=45, vjust=1, size=10, hjust=1),
        axis.ticks = element_blank(),
        axis.title.x = element_blank(), 
        legend.title = element_blank()) + 
   stat_summary(fun.data=mean_sdl, geom="pointrange", color="red")
```

![](Exploratory_analysis_of_datasets_files/figure-markdown_github-ascii_identifiers/violin%20plots-1.png)

``` r
ggplot(data = 
   df_sacros_visualization[which(df_sacros_visualization$group == "velocity"),], 
        aes(x=variable, y=value)) + 
   geom_violin(aes(fill=variable), draw_quantiles = c(0.25, 0.5, 0.75)) +
   scale_fill_brewer(palette="Pastel2") +
   theme_bw()+
   theme(axis.text.x=element_text(angle=45, vjust=1, size=10, hjust=1),
        axis.ticks = element_blank(),
        axis.title.x = element_blank(), 
        legend.title = element_blank()) + 
   stat_summary(fun.data=mean_sdl, geom="pointrange", color="red") 
```

![](Exploratory_analysis_of_datasets_files/figure-markdown_github-ascii_identifiers/violin%20plots-2.png)

``` r
ggplot(data = 
   df_sacros_visualization[which(df_sacros_visualization$group == 
                                   "acceleration"),], 
        aes(x=variable, y=value)) + 
   geom_violin(aes(fill=variable), draw_quantiles = c(0.25, 0.5, 0.75)) +
   scale_fill_brewer(palette="Set2") +
   theme_bw()+
   theme(axis.text.x=element_text(angle=45, vjust=1, size=10, hjust=1),
        axis.ticks = element_blank(),
        axis.title.x = element_blank(), 
        legend.title = element_blank()) + 
   stat_summary(fun.data=mean_sdl, geom="pointrange", color="red")
```

![](Exploratory_analysis_of_datasets_files/figure-markdown_github-ascii_identifiers/violin%20plots-3.png) To understand how correlated our inputs are, corellation heatmap with respective correlations is shown below.

``` r
df_sarcos_corr <- round(cor(df_sarcos[,1:21]),2)
df_sarcos_corr[upper.tri(df_sarcos_corr)] <- NA
df_sarcos_corr_melted <- reshape2::melt(df_sarcos_corr, na.rm = TRUE)


ggplot(data = df_sarcos_corr_melted, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", 
   name="Pearson\nCorrelation")+
  geom_raster() +
  theme_bw()+
  theme(axis.text.x=element_text(angle=45, vjust=1, size=10, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        #panel.background = element_blank(),
        axis.ticks = element_blank(),
        legend.justification = c(1, 0),
        legend.position = c(0.6, 0.7),
        legend.direction = "horizontal")+
  coord_fixed()+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                title.position = "top", title.hjust = 0.5)) + 
  geom_text(aes(Var1, Var2, label = value), color = "black", size = 2)
```

![](Exploratory_analysis_of_datasets_files/figure-markdown_github-ascii_identifiers/correlations-1.png)
