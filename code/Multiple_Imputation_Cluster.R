require(rstudioapi)
current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path ))
require(data.table)
require(mice)

# Data cleaning in R

# Import data
data = fread('combo2.csv', header=TRUE)
ind <- which(data[data$age > 110,])
data$age[ind] <- NA

ind2 <-which(data[data$dwelltype == 'Landlord',])
data$dwelltype[ind2] <- "House"

data <- apply(data, 2, function(x) gsub("^$|^ $", NA, x))
data <- as.data.frame(data)


# Multiple Imputation
df = subset(data, select = -c(zipcode,city,latitude,longitude) )

data <- df
as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}
data[,c("age","lenatres","nadults","nkids","premium","tenure","year")] <- lapply(data[,c("age","lenatres","nadults","nkids","premium","tenure","year")], as.numeric.factor)
data$credit <- factor(data$credit, ordered = TRUE,levels = c("low", "medium", "high"))
sapply(data, function(x) sum(is.na(x)))

data2 = data[1:1000,]
init = mice(data2, maxit=0) 
meth = init$method
predM = init$predictorMatrix

predM[, c("id", "train", "cancel")]=0
meth[c("id","train","cancel")]=""

set.seed(103)
start_time = proc.time()
imputed = mice(data2, method=meth, predictorMatrix=predM, m=1, maxit=1)
end_time = proc.time() - start_time
print(end_time)

imputedData_1 = complete(imputed,1)

fwrite(imputedData_1, file = '/imputed_12.csv', na=NA)
