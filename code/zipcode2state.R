require(rstudioapi)
current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path ))
require(data.table)
require(zipcode)
require(mice)


train <- fread("combo.csv")
train_zip <- train$zipcode
train_zip <- clean.zipcodes(train_zip)

train_state <- train_zip
train_latitude <- train_zip
train_longitude<- train_zip
  
data(zipcode)
state_list <- c("AZ","CO","IA","PA","VA","WA")
for (i in 1:length(state_list)){
  state <- state_list[i]
  code <- zipcode$zip[which(zipcode$state == state)]
  ind <- which(train_zip %in% code)
  train_state[ind] <- state
  print(i)
}

train$zipcode <- clean.zipcodes(train$zipcode)
test <- merge(train,zipcode,by.x = "zipcode",by.y="zip",all.x = TRUE)

write.csv(test,file="combo2.csv",row.names = FALSE)
