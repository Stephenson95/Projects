#Libraries
library(tidyverse)
library(corrplot)
#install.packages('Amelia')
library(Amelia)
library(gridExtra)
options(warn=-1)

#Importing data
data_file_path <- "C:\\Users\\Stephenson\\Desktop\\ANU\\Statistical Learning\\Final\\Data-20210527"
test_data <- read.csv(paste(data_file_path, '\\test.csv', sep = ""))
train_data <- read.csv(paste(data_file_path, '\\train.csv', sep = ""))

###EDA
fields_to_analyse <- c("Station", "Longitude", "Latitude", "Elev", "Drainage", "Day", 
                       "Conductivity", "Rainfall", "Stream.Discharge", "Stream.Water.Level", 
                       "Water.Temperature")
num_fields <- c("Longitude", "Latitude", "Elev", "Drainage", "Day", 
                "Conductivity", "Rainfall", "Stream.Discharge", "Water.Temperature")
continuous_fields <- num_fields[-c(5)]
cat_fields <- c("Station", "Stream.Water.Level")

for(col in cat_fields){#Convert categorical variables to factors
  if(col != 'Stream.Water.Level'){
    train_data[,col] <- as.factor(train_data[,col])
    test_data[,col] <- as.factor(test_data[,col])
  } else{
    train_data[,col] <- as.factor(train_data[,col])
  }
}

analyse_df <- train_data[fields_to_analyse]

summary(analyse_df[num_fields])

#Note Day min value is 366 and max is 6941 (19 years) because we need to predict the first
#and last years.

summary(analyse_df[cat_fields])

#Sanity check for "NA" or "NAN" entries
sapply(analyse_df, function(x) sum(is.na(x), is.nan(x)))

#Missing covariates

#Conductivity
investigation_conduc <- analyse_df %>% 
  group_by(is.na(Conductivity)) %>%
  summarise(avg_stream_discharge = mean(Stream.Discharge, na.rm=TRUE))

investigation_conduc <- analyse_df %>% 
  group_by(is.na(Conductivity)) %>%
  summarise(avg_stream_discharge = mean(as.numeric(Stream.Water.Level)-1, na.rm=TRUE))

#Water.Temparature
investigation_water <- analyse_df %>% 
  group_by(is.na(Water.Temperature)) %>%
  summarise(avg_stream_discharge = mean(Stream.Discharge, na.rm=TRUE))

investigation_water <- analyse_df %>% 
  group_by(is.na(Water.Temperature)) %>%
  summarise(avg_stream_discharge = mean(as.numeric(Stream.Water.Level)-1, na.rm=TRUE))

#Not drastically different, most likely missing at random. Imputation is an appropriate further approach.

#p2 <-investigation_water %>% 
#  ggplot(aes(y=reorder(Station, prop_missing_water), x=prop_missing_water)) + 
#  geom_bar(stat="identity") +
#  ylab('Station')

#grid.arrange(p1,p2, ncol = 2)

rm(p1,p2,investigation_conduc, investigation_water)

#Extract Stations with more than 90% missing
#missing_stations_conduc <- investigation_conduc[investigation_conduc$prop_missing_cond >0.9, 'Station']
#missing_stations_water <- investigation_water[investigation_water$prop_missing_water >0.9, 'Station']

#nrow(setdiff(missing_stations_conduc, missing_stations_water))

#nrow(intersect(missing_stations_conduc, missing_stations_water))/nrow(missing_stations_conduc)
#nrow(intersect(missing_stations_conduc, missing_stations_water))/nrow(missing_stations_water)

#rm(missing_stations_conduc, missing_stations_water)

#Looking at the correlation of numeric data 
cor_matrix<-cor(na.omit(analyse_df[num_fields]))
cor_matrix
corrplot(cor_matrix, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

rm(cor_matrix)

#Distributions of continuous variables
par(mfrow = c(2,4))
for(field in num_fields[-c(5)]){
  hist(analyse_df[,field], main = paste(field), breaks = 8)
}
par(mfrow = c(1,1))

#Box-Plot of continuous numerical against Stream.Water.Level
box_plot_variables <- function(x_variable, y_variable, x_unit, y_unit){
  boxplot(analyse_df[,y_variable]~analyse_df[,x_variable], main =paste(y_variable, 'vs', x_variable),
          ylab = paste(y_variable, y_unit), xlab = paste(x_variable, x_unit))
}

par(mfrow = c(2,4))
for(field in continuous_fields){
  box_plot_variables('Stream.Water.Level', field, '', '')
}
par(mfrow = c(1,1))

#Distribution of continuous variables with stream water level coloured
p1 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Longitude, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p2 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Latitude, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p3 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Elev, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p4 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Drainage, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p5 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Conductivity, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p6 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Rainfall, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p7 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Stream.Discharge, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
p8 <- ggplot(analyse_df[!is.na(analyse_df$Stream.Water.Level),], aes(x=Water.Temperature, fill=Stream.Water.Level, color=Stream.Water.Level)) +
  geom_histogram(position="identity", bins=8)
grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8, ncol = 4)

rm(p1,p2,p3,p4,p5,p6,p7,p8)

#Line graph of Days and average Stream.Water.Level (remove na's)
analyse_df %>%
  group_by(Day) %>%
  summarise(Avg_Stream_lvl = mean(as.numeric(Stream.Water.Level), na.rm=TRUE) - 1) %>%
  ggplot(aes(x=Day, y=Avg_Stream_lvl)) + 
  geom_line() +
  ylab('Proportion of High Stream Lvl')

#Category distribution of Station and Stream Water Level
investigation <- data.frame(table(analyse_df$Station, analyse_df$Stream.Water.Level))
names(investigation) <- c('Station', 'Stream.Water.Level', 'Frequency')
investigation <- investigation %>% 
  arrange(desc(Frequency), Station, Stream.Water.Level)
ggplot(investigation, aes(x=Frequency, y=reorder(Station, -Frequency), fill = Stream.Water.Level)) + 
  geom_bar(stat="identity") +
  ylab('Station')


#Scatter Plot of Stream.Discharge and numerical variables
scatter_plot_variables <- function(x_variable, y_variable, x_unit, y_unit) {
  plot(analyse_df[,x_variable], analyse_df[,y_variable], main = x_variable, 
       ylab = paste(y_variable, y_unit), xlab = paste(x_variable, x_unit))
  abline(lm(analyse_df[,y_variable]~analyse_df[,x_variable]), col = 'red')
}

par(mfrow = c(2,4))
for(field in continuous_fields[-c(7)]){
  scatter_plot_variables(field, 'Stream.Discharge', '', '')
}
par(mfrow = c(1,1))

#Stream.Discharge vs Stream Water Lvl
investigation <- analyse_df %>%
  filter(!is.na(Stream.Water.Level)) %>%
  group_by(Stream.Water.Level) %>%
  summarise(avg_discharge = mean(Stream.Discharge, na.rm=TRUE),
            med_discharge = median(Stream.Discharge, na.rm=TRUE))

data.frame(investigation)

#Stream.Discharge vs Station
investigation <- analyse_df %>%
  filter(!is.na(Stream.Water.Level)) %>%
  group_by(Station) %>%
  summarise(avg_discharge = mean(Stream.Discharge, na.rm=TRUE),
            med_discharge = median(Stream.Discharge, na.rm=TRUE))

ggplot(investigation, aes(x=med_discharge, y=reorder(Station, -med_discharge))) + 
  geom_bar(stat="identity") +
  ylab('Station') +
  xlab('Median Discharge')

#Examining Latitude & Longitude data
ggplot(data = train_data, aes(x = Longitude, y = Latitude, col = Station)) + geom_point()

###Modeling

#Imputation of missing values
set.seed(1)
imputed_train_data <- amelia(train_data, m = 3, ts = 'Day',ords = c('Station','Stream.Water.Level'), id_vars = c('Case.ID'))
imputed_test_data <- amelia(test_data, m = 3, ts = 'Day',ords = c('Station'), id_vars = c('Case.ID'))

mod_data <- train_data
testing_data <- test_data

#Feature Engineering
mod_data$Elev_bin <- ifelse(mod_data$Elev > 0, 'Elev > 0', 'Elev = 0')
mod_data$Elev_bin <- factor(mod_data$Elev_bin, levels = c('Elev = 0','Elev > 0'))
mod_data <- mod_data[-c(5)]

#PCA of latitude, longitude and Elevation of data
X <- mod_data[,c('Longitude', 'Latitude', 'Elev')]
mod.pc <- prcomp(X)
summary(mod.pc)
mod.pc

mod_data$Location_coords <- mod.pc$x[,'PC1']

#Drop Case ID
mod_data <-  mod_data[, -c(1)]
testing_data <- testing_data[, -c(1)]
#Drop Longitude and Latitude
mod_data <- mod_data[, -c(2,3)]
testing_data <- testing_data[, -c(2,3)]

#Predicting Stream Discharge

#Linear Regression Naive Model(Backward selection based on p-values)
min_stream_discharge_value <- min(analyse_df[analyse_df[,'Stream.Discharge']!=0 & !is.na(analyse_df[,'Stream.Discharge']),'Stream.Discharge'])
naive_mod_1 <- lm(log(Stream.Discharge + min_stream_discharge_value)~., data=mod_data[-c(2,9)])
summary(naive_mod_1)

naive_mod_2 <- lm(log(Stream.Discharge + min_stream_discharge_value)~1, data=mod_data[-c(2,9)])
summary(naive_mod_2)

#Linear Regression (Forward Selection)
#Performing 10 fold cross validation approach and using a probability threshold of 50%
set.seed(100)
K <- 10
size <- nrow(train_data)/K #Getting size based on 10 folds
overall_mse_df <- as.data.frame(matrix(0, ncol = 6, nrow = 3)) #Create df for out (only using 7 predictors)
names(overall_mse_df) <- seq(1:9)

# 1 covariate model =================================
mse_df <- as.data.frame(matrix(0, ncol = 8, nrow = 3)) #Create df for out
names(mse_df) <- fields_to_analyse[-c(2,3,9)]
rownames(mse_df) <- c('CV(k)', 'CV(k)-SE', 'CV(k)+SE')

#Using Station as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~Station, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Station'] <- result

#Using log(drainage) as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Drainage), data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Drainage'] <- result

#Using Day as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~Day, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Day'] <- result

#Using log(Conductivity + min(Conductivity) + 0.1) as a predictor
min_Conductivity_value <- min(mod_data$Conductivity[!is.na(mod_data$Conductivity) & mod_data$Conductivity !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Conductivity),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Conductivity-min_Conductivity_value+0.1), data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Conductivity'] <- result

#Using log(Rainfall + min(Rainfall)) as a predictor
min_rainfall_value <- min(mod_data$Rainfall[!is.na(mod_data$Rainfall) & mod_data$Rainfall !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value), data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Rainfall'] <- result


#Using Stream.Water.Level as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~Stream.Water.Level, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Stream.Water.Level'] <- result

#Using Water.Temperature as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Water.Temperature),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~Water.Temperature, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Water.Temperature'] <- result

#Using Elev as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~Elev_bin, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Elev'] <- result

#Model moving forward log(Stream.Discharge) ~ log(Rainfall + min(Rainfall))
overall_mse_df[,1] <-mse_df[,'Rainfall']



# 2 covariate model =================================
mse_df <- as.data.frame(matrix(0, ncol = 7, nrow = 3)) #Create df for out
names(mse_df) <- fields_to_analyse[-c(2,3,8,9)]
rownames(mse_df) <- c('CV(k)', 'CV(k)-SE', 'CV(k)+SE')

#Using Station as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Station'] <- result

#Using log(drainage) as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + log(Drainage), data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Drainage'] <- result

#Using Day as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Day, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Day'] <- result

#Using log(Conductivity + min(Conductivity) + 0.1) as a predictor
min_Conductivity_value <- min(mod_data$Conductivity[!is.na(mod_data$Conductivity) & mod_data$Conductivity !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Conductivity),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + log(Conductivity-min_Conductivity_value+0.1), data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Conductivity'] <- result

#Using Stream.Water.Level as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Stream.Water.Level, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Stream.Water.Level'] <- result

#Using Water.Temperature as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Water.Temperature),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Water.Temperature, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Water.Temperature'] <- result

#Using Elev as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Elev_bin, data=cv_train_data)
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Elev'] <- result

#Model moving forward log(Stream.Discharge) ~ log(Rainfall + min(Rainfall)) + Station)
overall_mse_df[,2] <-mse_df[,'Station']



# 3 covariate model =================================
mse_df <- as.data.frame(matrix(0, ncol = 6, nrow = 3)) #Create df for out
names(mse_df) <- fields_to_analyse[-c(1,2,3,8,9)]
rownames(mse_df) <- c('CV(k)', 'CV(k)-SE', 'CV(k)+SE')

#Using log(drainage) as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage), data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Drainage'] <- result

#Using Day as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + Day, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Day'] <- result

#Using log(Conductivity + min(Conductivity) + 0.1) as a predictor
min_Conductivity_value <- min(mod_data$Conductivity[!is.na(mod_data$Conductivity) & mod_data$Conductivity !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Conductivity),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Conductivity-min_Conductivity_value+0.1), data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Conductivity'] <- result

#Using Stream.Water.Level as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + Stream.Water.Level, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Stream.Water.Level'] <- result

#Using Water.Temperature as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Water.Temperature),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + Water.Temperature, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data  
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Water.Temperature'] <- result

#Using Elev as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + Elev_bin, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data  
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Elev'] <- result

#Model moving forward log(Stream.Discharge) ~ log(Rainfall + min(Rainfall)) + Station + log(Drainage))
overall_mse_df[,3] <-mse_df[,'Drainage']




# 4 covariate model =================================
mse_df <- as.data.frame(matrix(0, ncol = 5, nrow = 3)) #Create df for out
names(mse_df) <- fields_to_analyse[-c(1,2,3,5,8,9)]
rownames(mse_df) <- c('CV(k)', 'CV(k)-SE', 'CV(k)+SE')

#Using Day as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Day, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Day'] <- result

#Using log(Conductivity + min(Conductivity) + 0.1) as a predictor
min_Conductivity_value <- min(mod_data$Conductivity[!is.na(mod_data$Conductivity) & mod_data$Conductivity !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Conductivity),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + log(Conductivity-min_Conductivity_value+0.1), data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Conductivity'] <- result

#Using Stream.Water.Level as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Stream.Water.Level, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Stream.Water.Level'] <- result

#Using Water.Temperature as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Water.Temperature),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Water.Temperature, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data  
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Water.Temperature'] <- result

#Using Elev as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data  
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Elev'] <- result

#Model moving forward log(Stream.Discharge) ~ log(Rainfall + min(Rainfall)) + Station + log(Drainage) + Elev)
overall_mse_df[,4] <-mse_df[,'Elev']



# 5 covariate model =================================
mse_df <- as.data.frame(matrix(0, ncol = 4, nrow = 3)) #Create df for out
names(mse_df) <- fields_to_analyse[-c(1,2,3,4,5,8,9)]
rownames(mse_df) <- c('CV(k)', 'CV(k)-SE', 'CV(k)+SE')

#Using Day as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + Day, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Day'] <- result

#Using log(Conductivity + min(Conductivity) + 0.1) as a predictor
min_Conductivity_value <- min(mod_data$Conductivity[!is.na(mod_data$Conductivity) & mod_data$Conductivity !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Conductivity),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + log(Conductivity-min_Conductivity_value+0.1), data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Conductivity'] <- result

#Using Stream.Water.Level as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + Stream.Water.Level, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Stream.Water.Level'] <- result

#Using Water.Temperature as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Water.Temperature),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + Water.Temperature, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data  
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Water.Temperature'] <- result

#Model moving forward log(Stream.Discharge) ~ log(Rainfall + min(Rainfall)) + Station + log(Drainage) + Elev + Stream.Water.Level
overall_mse_df[,5] <-mse_df[,'Stream.Water.Level']



# 6 covariate model =================================
mse_df <- as.data.frame(matrix(0, ncol = 3, nrow = 3)) #Create df for out
names(mse_df) <- fields_to_analyse[-c(1,2,3,4,5,8,9,10)]
rownames(mse_df) <- c('CV(k)', 'CV(k)-SE', 'CV(k)+SE')

#Using Day as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + Stream.Water.Level + Day, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Day'] <- result

#Using log(Conductivity + min(Conductivity) + 0.1) as a predictor
min_Conductivity_value <- min(mod_data$Conductivity[!is.na(mod_data$Conductivity) & mod_data$Conductivity !=0])
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + Stream.Water.Level + log(Conductivity-min_Conductivity_value+0.1), data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Conductivity'] <- result

#Using Water.Temperature as a predictor
mse <- rep(0,K)
for(k in 1:K){
  #Initiate folds for data sampling
  fold <- sample(rep(1:K, each = size))
  cv_train_data <- mod_data[fold != k,]
  cv_test_data <- mod_data[fold == k,]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Discharge),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Rainfall),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Drainage),]
  cv_test_data <- cv_test_data[!is.na(cv_test_data$Stream.Water.Level),]
  #Fit model and predict on test data
  mod.train <- lm(log(Stream.Discharge + min_stream_discharge_value)~log(Rainfall + min_rainfall_value) + Station + log(Drainage) + Elev_bin + Stream.Water.Level + Water.Temperature, data=cv_train_data)
  mod.train$xlevels[['Station']] <- union(mod.train$xlevels[['Station']], levels(cv_test_data$Station)) #Relevel Station based on testing data
  pred.test <- sapply(exp(predict(mod.train, newdata = cv_test_data)) - min_stream_discharge_value, function(x) max(0,x))
  #Calculate mse
  mse[k] <- sum((cv_test_data$Stream.Discharge - pred.test)^2)/nrow(cv_test_data)
}
stderr_cv <- sqrt(sum((mse - mean(mse))^2)/(K-1))/sqrt(K)
result <- c(mean(mse), 
            mean(mse) - stderr_cv,
            mean(mse) + stderr_cv)
mse_df[,'Water.Temperature'] <- result

#Model moving forward log(Stream.Discharge) ~ log(Rainfall + min(Rainfall)) + Station + log(Drainage) + Elev + Stream.Water.Level + Day
overall_mse_df[,6] <-mse_df[,'Day']


#Plot MSE estimates
plot(1:6, overall_mse_df[1,], 
     type='o', lwd=3, col='red',
     ylab='MSE',
     main = 'No. of predictors vs MSE')
#     ylim = c(0.12,0.3))
lines(1:6, overall_mse_df[2,], 
      type='l', lty=2, col='red')
lines(1:6, overall_mse_df[3,], 
      type='l', lty=2, col='red')
legend('topleft', legend=c("Estimated CV(10)", "SE Intervals"),
       col=c("red", "red"), lty=c(1,2))

#We choose p=6 (condition~cp+log(trestbps)+thal+thalach+sex+age)as our final model

#Ridge Regression

#Lasso Regression

