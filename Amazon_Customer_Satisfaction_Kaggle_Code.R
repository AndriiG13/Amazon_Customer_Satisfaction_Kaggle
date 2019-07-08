library(tidyverse)
library(tidytext)
library(glmnet)
library(ROCR)


# Data import -------------------------------------------------------------

amazon <-  read_csv("bda18-amazon/amazon_baby.csv")
head(amazon)

# get index of training data (the ones that are non-NA values)
trainidx <-  !is.na(amazon$rating)
table(trainidx)


# Preprocessing -----------------------------------------------------------
#extract single words 
reviews_words <- amazon %>%
  mutate(id = row_number()) %>%
  unnest_tokens(token, review) %>%
  count(id, name, rating, token)

head(reviews)

# extract bigrams
reviews_2ngram <- amazon %>%
  mutate(id = row_number()) %>%
  unnest_tokens(token, review, token = "ngrams", n = 2) %>%
  count(id, name, rating, token)

head(reviews_2ngram)

# extract all features for wards
features_words <- 
  reviews_words %>%
  group_by(id) %>%
  mutate(nwords = sum(n)) %>% # the number of tokens per document
  group_by(token) %>%
  mutate(
    docfreq = n(), # number of documents that contain the token
    idf = log(nrow(amazon) / docfreq), # inverse document frequency ('surprise')
    tf = n / nwords, # relative frequency of token within the given document
    tf_idf = tf*idf
  ) %>%
  ungroup()

# extract all features for bigrams
features_2ngrams <- 
  reviews_2ngram %>%
  group_by(id) %>%
  mutate(nwords = sum(n)) %>% # the number of tokens per document
  group_by(token) %>%
  mutate(
    docfreq = n(), # number of documents that contain the token
    idf = log(nrow(amazon) / docfreq), # inverse document frequency ('surprise')
    tf = n / nwords, # relative frequency of token within the given document
    tf_idf = tf*idf
  ) %>%
  ungroup()

# put into sparse matrix for words
dtm_words <- 
  filter(features_words, docfreq > 10) %>%
  cast_sparse(row=id, column=token, value = tf_idf)

dim(dtm_words)

# put into sparse matrix for bigrams
dtm_2ngrams <-
  filter(features_2ngrams, docfreq > 10) %>%
  cast_sparse(row=id, column=token, value = tf_idf)

dim(dtm_2ngrams)

# checking which observations may have been ommiited and creating the train index 
#(when switching betwen single words and bigrams the "trainidx" object needs to be removed and 
#this script ran again)
used_rows = as.integer(rownames(dtm_2ngrams))
used_amazon = amazon[used_rows, ]
trainidx = trainidx[used_rows]
table(trainidx)


# Making the response variable --------------------------------------------
# extract ratings
y = used_amazon$rating

# make logical
y_train <- ifelse(y[trainidx] > 3, "1", "0")

# factorize
y_train <- factor(y_train)


# Fitting models ----------------------------------------------------------
##We decided to stick with the glmnet package as it is very good with dealing with sparse data
#and handling large number of predictors.
##Specifically what w did was run a cross validated glmnet for both lasso and ridge models and then
##we extracted the most optimal lambda and lambda that is within 1 standard error for each ridge and
#lasso. Resulting in total of 4 models for words and 4 models for bigrams.


##################################### First for words
# fit a cross-validated lasso glmnet 
fit_lasso_cv_word <- cv.glmnet(dtm_words[trainidx, ], y_train, 
                                 family = "binomial", type.measure = 'auc', alpha = 1)
# plot lambdas against AUC
plot(fit_lasso_cv_word, xvar = "lambda")

# extract the optimal lambda and optimal lambda within 1SE
best_l_lasso_word <- fit_lasso_cv_word$lambda.min

best_l_lasso_1se_word <- fit_lasso_cv_word$lambda.1se



# fit a cross-validated ridge glmnet 
fit_ridge_cv_word <- cv.glmnet(dtm_words[trainidx, ], y_train, 
                               family = "binomial", type.measure = 'auc', alpha = 0)
# plot lambdas against AUC
plot(fit_ridge_cv_word, xvar = "lambda")

# extract the optimal lambda and optimal lambda within 1SE
best_l_ridge_word <- fit_ridge_cv_word$lambda.min

best_l_ridge_1se_word <- fit_ridge_cv_word$lambda.1se



##################################### Now for bigrams
# fit a cross-validated lasso glmnet 
fit_lasso_cv_bigram <- cv.glmnet(dtm_2ngrams[trainidx, ], y_train, 
                          family = "binomial", type.measure = 'auc', alpha = 1)
# plot lambdas against AUC
plot(fit_lasso_cv_bigram, xvar = "lambda")

# extract the optimal lambda and optimal lambda within 1SE
best_l_lasso_bg <- fit_lasso_cv_bigram$lambda.min

best_l_lasso_1se_bg <- fit_lasso_cv_bigram$lambda.1se



# fit a cross-validated cv.glmnet 
fit_ridge_cv_bg <- cv.glmnet(dtm_2ngrams[trainidx, ], y_train, 
                          family = "binomial", type.measure = 'auc', alpha = 0)
# plot lambdas against AUC
plot(fit_ridge_cv_bg, xvar = "lambda")

# extract the optimal lambda and optimal lambda within 1SE
best_l_ridge_bg <- fit_ridge_cv_bg$lambda.min

best_l_ridge_bg <- fit_ridge_cv_bg$lambda.1se


# Plotting ROC curves ---------------------------------------------
##In between on running the models we decided to plot  ROC curves just to check we were
#on the right path. Glmnet does not provide a direct way to do this so we had to use ROCR package

# making predictions based on the best lambda and lasso model 
roc_pred = tibble(id = which(trainidx), rating = y_train) %>%
  mutate(pred = predict(fit_ridge_cv_bg, dtm_2ngrams[trainidx, ], s = best_l_ridge_bg, type = "response")) # add predictions

#plotting a quick ROC curve just to check 
roc_pred <- prediction(roc_pred$pred, roc_pred$rating)
perf <- performance(pred_best,"tpr","fpr")
performance(pred_best, "auc") # shows calculated AUC for model
plot(perf, colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )




# Predictions on test set for submission ----------------------------------

sample_submission <-  read_csv("bda18-amazon/amazon_baby_testset_sample.csv")
sample_submission$Id <- as.character(sample_submission$Id)

# used_rows computed earlier contains all the indices of reviews used in dtm
all_test_reviews = which(is.na(amazon$rating))
missing_test_ids = setdiff(used_rows, all_test_reviews)

# best prediction if no review features are available
best_default_prediction = mean(as.integer(as.character(y_train))) 
cat("best baseline prediction:", best_default_prediction,"\n")

##The model which gave us the best prediction was the lasso model on bigrams with the optimal
#lambda
dtm_test_predictions <- 
  tibble(Id = as.character(used_rows[!trainidx]),
         pred=predict(fit_lasso_cv_bigram, dtm_2ngrams[!trainidx, ], s = fit_lasso_cv_bigram, 
                      type = "response")[,1]
  )


pred_df = sample_submission %>%
  left_join(dtm_test_predictions) %>%
  mutate(Prediction = ifelse(Id %in% missing_test_ids, best_default_prediction, pred)) # add predictions
#nrow(pred_df)
head(pred_df)

##replace possible NAs with the best default prediction
pred_df[is.na(pred_df)]  <- best_default_prediction

##check no NAs remain
which(is.na(pred_df$Prediction))

##make the final prediction and write the file 
pred_df %>%
  transmute(Id=Id, Prediction = Prediction) %>%
  write_csv("my_submission7.csv")
file.show("my_submission7.csv")


