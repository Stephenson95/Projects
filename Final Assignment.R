#Importing data
data_file_path <- "C:\\Users\\Stephenson\\Desktop\\ANU\\Statistical Learning\\Final\\Data-20210527"
test_data <- read.csv(paste(data_file_path, '\\test.csv', sep = ""))
train_data <- read.csv(paste(data_file_path, '\\train.csv', sep = ""))

#EDA
fields_to_analyse <- c("Station", "Longitude", "Latitude", "Elev", "Drainage", "Day", 
                       "Conductivity", "Rainfall", "Stream.Discharge", "Stream.Water.Level", 
                       "Water.Temperature")
num_fields <- c("Longitude", "Latitude", "Elev", "Drainage", "Day", 
                "Conductivity", "Rainfall", "Stream.Discharge", "Water.Temperature")
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

summary(analyse_df[cat_fields])

#Sanity check for "NA" or "NAN" entries
sapply(analyse_df, function(x) sum(is.na(x), is.nan(x)))

#Looking at the correlation of numeric data 
library(corrplot)
cor_matrix<-cor(na.omit(analyse_df[num_fields]))
cor_matrix
corrplot(cor_matrix, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

#Distributions


#Co-variate Plot


