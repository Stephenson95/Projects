library(MASS)
library(foreign)

import_path <- "C:\\Users\\Stephenson\\Desktop\\Code\\ASD"

dat <- read.csv(paste(import_path, '\\cleaned dataset.csv', sep = ""))

dat$GoldsteinScale <- as.factor(dat$GoldsteinScale)

m <- polr(GoldsteinScale ~ QuadClass_2 + QuadClass_3 + QuadClass_4 + PCA_publicity + AvgTone, data = dat, ologit = 'or')

summary(m)

ci <- confint(m)

ci

exp(cbind(OR = coef(m), ci))

 
