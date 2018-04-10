install.packages("RCurl")
library(RCurl)
library(tm)
library(NLP)
test_data_url <- "https://storage.googleapis.com/kaggle-competitions-data/kaggle/2558/testdata.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1523528165&Signature=V8bIoX%2Fq7QX6AgKwWATe1LlUXudaeqgVuOmXnEKB9NlXlwizrexHh4WY%2FropYcZvrSBhVoXDgUhxs%2BsRuxchmrjMczxg7sxiSS%2BC8HazfWFkO9oDZWhILCNet1JPQGFszk5i7pwJWtUaQXzgGPpj9U97BZnfy0lDeUV5jOOrr3qiP2fHfGTu5QFDgcMVynVV3FYvnEkDPuws7iGiBuFi%2Bbs9hx5TsB60EZWMO%2B8peJt7cJbJUJvJU9EKFj5oeAg5KEWn83dve710IO%2BAe5DyQt9gq5kJa7T552E5syt5dDrye0b9Te5YBS2Z1pTzJgQkcNvG4FS9lyPzjzpfX3LTQg%3D%3D"
train_data_url <- "https://storage.googleapis.com/kaggle-competitions-data/kaggle/2558/training.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1523528803&Signature=Q9YUol7b0L3x9sp9hEtLuGrHjhtY49%2F4Wm77QmFVjIgzeUKSVXGWr3ImjduFceXaudNWdWBGco125CQhkc8xhftCvxYE1z3wVIlZxAy570G7SPFapF%2FDKjfjXJqEUcGHazEJEzbL7%2FnY1wITh0iDNZWhRltk7DE80LCbY8mIlelDHEOBPVKCss5x18ym6ax7Qa1m8tcKfFxSl6Jz7UL4GJqnrYGoHztGXJ6cso8k4n3MjcdHXq9zfVqOKYV602T83ADGBv25a7exRUZlo%2BNhIkJCLA%2BEdCOtMwgdF5DIQ3C7IeDpqtmzlTIhXlfPBtCfwIH8mIrpow3l0YNBvh5%2FLQ%3D%3D"
test_data_file <- getURL(test_data_url)
train_data_file <- getURL(train_data_url)
train_data_df <- read.csv(
  text = train_data_file, 
  sep='\t', 
  header=FALSE, 
  quote = "",
  stringsAsFactor=F,
  col.names=c("Sentiment", "Text"))
test_data_df<-read.csv(
  text = test_data_file, 
  sep='\t', 
  header=FALSE, 
  quote = "",
  stringsAsFactor=F,
  col.names=c("Text"))
train_data_df$Sentiment <- as.factor(train_data_df$Sentiment)
head(train_data_df)
table(train_data_df$Sentiment)
mean(sapply(sapply(train_data_df$Text, strsplit, " "), length))
corpus<-Corpus(VectorSource(c(train_data_df$Text,test_data_df$Text)))
corpus[1]$content
corpus <- tm_map(corpus, tolower)
corpus<-tm_map(corpus,removePunctuation)
corpus<-tm_map(corpus,removeWords,stopwords(kind = "en"))
corpus<-tm_map(corpus,stripWhitespace)
corpus<-tm_map(corpus,stemDocument)
corpus[1]$content
library(SnowballC)
dtm <- DocumentTermMatrix(corpus)
dtm
sparse <- removeSparseTerms(dtm, 0.99)
sparse
important_words_df <- as.data.frame(as.matrix(sparse))
colnames(important_words_df) <- make.names(colnames(important_words_df))
important_words_train_df <- head(important_words_df, nrow(train_data_df))
important_words_test_df <- tail(important_words_df, nrow(test_data_df))
train_data_words_df <- cbind(train_data_df, important_words_train_df)
test_data_words_df <- cbind(test_data_df, important_words_test_df)
train_data_words_df$Text <- NULL
test_data_words_df$Text <- NULL
library(caTools)
set.seed(1234)
spl <- sample.split(train_data_words_df$Sentiment, .85)
eval_train_data_df <- train_data_words_df[spl==T,]
eval_test_data_df <- train_data_words_df[spl==F,]

log_model <- glm(Sentiment~., data=eval_train_data_df, family=binomial)
summary(log_model)
log_pred <- predict(log_model, newdata=eval_test_data_df, type="response")
table(eval_test_data_df$Sentiment, log_pred>.5)
(453 + 590) / nrow(eval_test_data_df)
log_pred_test <- predict(log_model, newdata=test_data_words_df, type="response")
test_data_df$Sentiment <- log_pred_test>.5
set.seed(1234)
spl_test <- sample.split(test_data_df$Sentiment, .0005)
test_data_sample_df <- test_data_df[spl_test==T,]
test_data_sample_df[test_data_sample_df$Sentiment==T, c('Text')]
test_data_sample_df[test_data_sample_df$Sentiment==F, c('Text')]
submit<-data.frame(Sentiment=test_data_df$Sentiment,Text=test_data_df$Text)
write.csv(submit,file="firstsubmission",row.names = FALSE)
