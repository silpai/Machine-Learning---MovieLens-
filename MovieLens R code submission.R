##########################################################
#title: "MovieLens Project - HarvardX:PH125.9x Data Science Capstone"
#author: "Silvane Paixao (silpai)"
#date: "1/2/2021"
##########################################################

##########################################################
# Notes:
#clear unusued memory and increase memory limit
#invisible(gc())
#1 this code requires a computer with high memory ram (greater than 8GB would be ideal).
#2.codes will take  few minutes to run and will be slow to return the results
#3. This project was running with memory.limit(size = 70000). You may need to increase your memory.limit()
  #In RStudio, to increase memory: 
    #file.edit(file.path("~", ".Rprofile"))
  #then in .Rprofile type this and save
    #invisible(utils::memory.limit(size = 70000))
    
##########################################################


############################################################################################
                     # Time for the code to run - please be patience
                        #Total estimated time: 80 min (1h40m)
# From beginning to predictions: 40min
# Matrix factorization: 40min 
############################################################################################

############################################################################################
#clear unusued memory and increase memory limit
#invisible(gc())
#invisible(memory.limit(size = 70000))
############################################################################################
#tinytex::install_tinytex()                                                                           # LaTeX requirement



if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")       # To extract data
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")         # To convert timestamp
if(!require(stringi)) install.packages("stringi", repos = "http://cran.us.r-project.org")             # To extract movie year premier
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org") 
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org") 
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")   # To custom color pallets
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")       # To format tables
if(!require(recosystem))  install.packages("recosystem", repos = "http://cran.us.r-project.org")      # To create matrix factorization

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(stringi)
library(ggthemes)
library(wordcloud)
library(RColorBrewer)
library(kableExtra)
library(recosystem)

#############################################    Sample code from EdX ############################################# 

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))



ratings <- as.data.frame(ratings) %>% mutate(movieId = as.numeric(movieId),
                                                 userId = as.numeric(userId),
                                                 rating = as.numeric(rating),
                                                 timestamp = as.numeric(timestamp))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

############################################# 1. Introduction  #############################################

#Recommendation systems rely on the notion of similarity between objects. Nowadays it has been applied to recommend movies, books, hotels, restaurants, doctors, etc. According to Forte and Miller (2017) customers can be considered like each other if they share most of the products that they have purchased, and items can be considered similar to each other if they share a large number of customers who purchased them.

#In the real-world, Recommendation systems have been used by e-commerce websites such as Amazon.com, Netflix, iTunes to improve revenue. Customers are guided to the right product and receive recommendations on products with similar characteristics. 

#This project is a requirement for the HarvardX Professional Certificate Data Science Program which aims to predict movies ratings using *"MovieLens 10M Dataset"*.



#############################################  2. Methodology and Analysis ############################################# 

# 1. Several data transformation was implemented to create new categories for:
  # movie rating decade - transformed timestamp into year and after categorizing the ratings in the 1990s and 2000s
  # release decade: split the title and retrieve the release year, then categorize into decades from 1910s to 2000s
  # rating group: if the star rating was half or whole star
  # single genre: remove all the genres combination
  # date: timestamp was also transformed into YYYY-MM-DD
  # Validations were preformed to verify if the data wrangling was implemented correctly
# 2. Exploratory Analysis (EDA) were developed to get more insights for the recommendation system and have a better  understanding of the data
# 3. Modelling was used to create the predictions:
  # Random prediction, 
  # Linear Models: Linear regression and correlation using edx, followed by predictions using train_set to create the model for Only mean of the ratings,  Adding Movie effects (bi), Adding Movie + User Effect (bu), Adding Movie + User + time Effect (bt) . Both test_set and validation were used to predict and calculate the RMSE
  #	Matrix factorization (recosystem)

#Final metric is the root mean square estimate (RMSE).  A lower RMSE means a better, more accurate prediction. According to Perrier (2017) RMSE is defined as the sum of the squares of the difference between the real values and the predicted values. The closer the predictions are to the real values, the lower the RMSE is.
#A lower RMSE means a better, more accurate prediction. The aim for successful metrics is to reach RMSE Capstone target < *0.8649*.

#############################################  2.1. edx Data Transformation ############################################# 
#During the process of data transformation, a series of validations occurred to ensure retrieving the correct data. For example, after splitting the title to get the release year, it was found that the generated premier_date retrieve values were between 1000 and 9000. 
#It was clear that the split numbers related to the title were retrieved as release year. Re-coding was necessary to fix this issue.

#1.augment edx dataframe
edx_transf<-edx %>% mutate(year_rated = year(as_datetime(timestamp)),                                            # transform timestamp to year
                    premier_date=as.numeric(stri_extract(title, regex = "(\\d{4})", comments = TRUE)))           # extract date from title
                 
head(edx_transf)


####### a) Validate initial edx_transf transformations #############################

#1. Validate transformation (split of the title into premier_date)
start_premier_date=min(edx_transf$premier_date)             # movies release start year 
end_premier_date=max(edx_transf$premier_date)               # movies release end year

#2.Identify release year. It was found that title split generated premier_date that was in between 1000 and 9000. It was clear that during the split numbers related to the title were retrieved as release year
premier_year_tofix <- edx_transf %>% group_by(movieId) %>% 
  filter (premier_date<1910|premier_date>2009) %>% 
    select(movieId, title, premier_date)
unique(premier_year_tofix)

#3. Recode the 17 release years between 1000 and 9000
edx_transf[edx_transf$movieId == "2308", "premier_date"] <- 1973
edx_transf[edx_transf$movieId == "5310", "premier_date"] <- 1985
edx_transf[edx_transf$movieId == "671", "premier_date"] <- 1996
edx_transf[edx_transf$movieId == "4159", "premier_date"] <- 2001
edx_transf[edx_transf$movieId == "27266", "premier_date"] <- 2004
edx_transf[edx_transf$movieId == "8864", "premier_date"] <- 2004
edx_transf[edx_transf$movieId == "2311", "premier_date"] <- 1984
edx_transf[edx_transf$movieId == "5472", "premier_date"] <- 1972
edx_transf[edx_transf$movieId == "4311", "premier_date"] <- 1998
edx_transf[edx_transf$movieId == "1422", "premier_date"] <- 1997
edx_transf[edx_transf$movieId == "8905", "premier_date"] <- 1992
edx_transf[edx_transf$movieId == "53953", "premier_date"] <- 2007
edx_transf[edx_transf$movieId == "6645", "premier_date"] <- 1971
edx_transf[edx_transf$movieId == "6290", "premier_date"] <- 2003
edx_transf[edx_transf$movieId == "8198", "premier_date"] <- 1960
edx_transf[edx_transf$movieId == "2691", "premier_date"] <- 1998
edx_transf[edx_transf$movieId == "26359", "premier_date"] <- 1976


#4. Implement final transformations
edx <-edx_transf %>% mutate(date = round_date(as_datetime(timestamp), unit = "month"),                                                              # transform timestamp into YYYY-MM-DD
                            movie_rated_era=ifelse (year_rated<=1999, "Ratings during 1990s","Ratings during 2000s"),                               # define the decades of the rated movies: 1990s (1990-1999) 2000s (2000-2009)        #extract date from title
                            premier_year_era=ifelse (premier_date >=1910 & premier_date <=1919, "1910s",                                            # define the decades of the premier date
                                              ifelse(premier_date >=1920 & premier_date <=1929, "1920s", 
                                               ifelse(premier_date >=1930 & premier_date <=1939, "1930s",
                                                ifelse(premier_date >=1940 & premier_date <=1949, "1940s",
                                                 ifelse(premier_date >=1950 & premier_date <=1959, "1950s",
                                                  ifelse(premier_date >=1960 & premier_date <=1969, "1960s",
                                                   ifelse(premier_date >=1970 & premier_date <=1979, "1970s",
                                                    ifelse(premier_date >=1980 & premier_date <=1989, "1980s",
                                                     ifelse(premier_date >=1990 & premier_date <=1999, "1990s",
                                                      ifelse(premier_date >=2000 & premier_date <=2009, "2000s","2010s")))))))))),
                            rating_group= ifelse((rating == 1 |rating == 2 | rating == 3 | rating == 4 | rating == 5), "whole star", "half star"),   # Define if rating are whole or half starts
                            Orig_genres=genres)  %>%                                                                                                 # keep the original genres column
                      separate(genres,c("single_genres"),sep = "\\|")                                                                                # split combination genres into single genres
head(edx)
#str(edx)



############################################# 2.2. Exploratory Analysis (EDA) ############################################# 

#### *Overall edx description* ####

#*MovieLens* is comprised by 2 datasets: *movies.dat* containing Movie ID, title and genres and *ratings.dat* containing User ID, Movie ID, rating, and timestamp.
#EDA indicated that edx contains 9,000,055 ratings applied to 106,770 movies from 797 genres combinations. Movies were rated by 69,878 users from 1995 to 2009. 
#Movie's premier took place between 1915 and 2008. 


Overall_edx_description <- data.frame(ratings_n=nrow(edx),      # count of number of rows
           unique_movies=length(unique(edx$movieId)),           # count of unique movieid
           unique_genres=length(unique(edx$Orig_genres)),       # count of unique genres
           users_n =length(unique(edx$userId)),                 # count of unique users
           start_year_rate=min(edx$year_rated),                 # when movies started to be rated by users
           end_year_rate=max(edx$year_rated),                   # when movies finished to be rated by users
           start_premier_date=min(edx$premier_date),            # movies release start year 
           end_premier_date=max(edx$premier_date))              # movies release end year
Overall_edx_description 

mytheme <- theme_minimal() + theme(panel.grid = element_blank(),axis.title = element_blank())       # Define minimal theme 


#### *Genres* ####

# After splitting the genre combinations (such as Action from "Action|Crime|Thriller" or Comedy from "Comedy|Romance") into single genres, 20 unique single genres were found.

top_genres <- edx %>% group_by(single_genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
top_genres

layout(matrix(c(1,2), nrow =2) , heights = c(1,4))                                 # create wordcloud based on the counts of the 20 unique single genres    
par(mar=rep(0,4))
plot.new()
text(x=0.5,y=0.5, "Top Single Genres by number of ratings")
wordcloud(words=top_genres$single_genres,freq=top_genres$count,min.freq=14,
          max.words = 19,random.order=FALSE,random.color=FALSE,
          rot.per=0.50,colors = brewer.pal(8,"Dark2"),scale=c(5,.2),
          family="plain",font=2,
          main = "Top single_genres by number of ratings")


#### Movies per single genre ####
single_genres_sum<-edx %>% group_by(single_genres,movieId) %>%
  summarize(n_rating_of_movie = n(), 
            mu_movie = mean(rating),
            sd_movie = sd(rating)) 
head(single_genres_sum)

#It was possible to identify that the top 3 single genres were Action, Comedy and Drama. Having Action and Comedy similar ratings. The lowest rated single genres were Romance, War and IMAX.

single_genres_sum %>%
  ggplot( aes(x=single_genres, y=n_rating_of_movie)) +
  geom_segment( aes(xend=single_genres, yend=0)) +
  geom_point( size=4, color="orange") +
  coord_flip() +   ggtitle("Movies per single genre") +
  labs(fill = "number of ratings", subtitle="How genre is distributed?", caption ="Note: genre combinations were removed") +
  mytheme + theme(axis.text.x = element_text(size = 10),
                  axis.text.y = element_text(size = 10),
                  legend.position = "none")

#### *Release Decade* ####

#The highest ratings were given to the 1980s and 1990s movies (3 and 4 rating stars). 
#Movies from the 1910s to 1970s also received good ratings, the majority were above 3 stars, having their highest number of ratings with 4 stars.

edx %>% group_by (rating,premier_year_era)%>%
  count(premier_year_era) %>% 
  ggplot(aes(x = premier_year_era, y = n, fill= n)) +  # Plot with values on top
  geom_bar(stat = "identity") +
  ggtitle("Release Decade") +
  labs(fill = "number of ratings", subtitle="How release decades are distributed over the star ratings?", caption ="Note: release decade category derived from the split of title") +
  mytheme + 
  theme(axis.text.y = element_blank(), axis.text.x =element_text(size = rel(0.75),angle = 90), legend.position = "bottom", legend.spacing.x = unit(1.0, 'cm'),legend.text = element_text(margin = margin(t = 10)))+
  guides(fill = guide_colorbar(title = "number of ratings",                                         # edit the legend bar
                               label.position = "bottom", label.hjust = 1,
                               title.position = "left", title.vjust = 0.75, 
                               frame.colour = "black",                                              # draw border around the legend
                               barwidth = 13,
                               barheight = 0.75)) +
  scale_fill_distiller(palette = "Spectral") +
  facet_wrap(~ rating,ncol=3)

# The analysis of the trend: "average ratings vs. date" shows that there is some evidence of a time effect. The movies from the 1970s and earlier received on average higher star rate score than movies which premier took place after the 1980s. 
#It can be noticed a decrease on the average star rating after year 2000 for the 1900s and 2000s  movies (from an average of about 4 to 3.5 stars), whereas the 1980s movies maintained the 3.5 star rating over the period of time. 

release_decade<-edx %>% 
  mutate(premier_year_era_group=ifelse (premier_date >=1910 & premier_date <=1979, "1970s or earlier",                                 # define the decades of the premier date : prior 1980s, 1980s,1990s,2000s
                                 ifelse(premier_date >=1980 & premier_date <=1989, "1980s",
                                  ifelse(premier_date >=1990 & premier_date <=1999, "1990s",
                                   ifelse(premier_date >=2000 & premier_date <=2009, "2000s","2010s"))))) %>%
  group_by(date, premier_year_era_group) %>%
  summarize(avgrating = mean(rating))
head(release_decade)

release_decade%>%
  ggplot(aes(date, avgrating, colour = premier_year_era_group)) +
  geom_point() +
  geom_smooth() +
 theme_bw() + theme(panel.grid = element_blank(),axis.title = element_blank()) +
  labs(colour = "Release decades", title="Timestamp (unit in month)", subtitle="Do movies prior 1980s recieve more star ratings than recent movies?")
 
       
  

  #############################################  2.3 Modelling  #############################################  
 #In this step edx was split into train_set (90% of the data) and test_set (10% of the data). Using the original code as template. Dimensions were: edx train_set = 8100048 x 12; edx test_set = 899990 X 12.
  
 #### Split edx into train_set and test_set ####

  set.seed(1, sample.kind="Rounding")
  test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
  train_set <- edx[-test_index,]
  temp <- edx[test_index,]

  # Make sure userId and movieId in test set are also in train set
  test_set <- temp %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId") 
 
 # Add rows removed from test set back into train set
  removed <- anti_join(temp, test_set)
  train_set <- rbind(train_set, removed)
  
  rm(test_index, temp, removed)
  
  
  validation<-validation %>% mutate(date = round_date(as_datetime(timestamp), unit = "month"))
  dim(validation)
  head(validation)
  
##############################################
  
  ####  Modelling #### 
  
#### 2.3.1 - Random Prediction #### 
# As the initial insight benchmark, probability distribution of the variables was calculated. This means that any model should be better than the values found. 
# EDA analysis identified that there were more movies being scored with 3- and 4-star ratings (23% and 28% respectively). This represents an overall 51% of the ratings given to just two score stars. 
# With the random prediction, ratings use the observed probabilities. Initially the probability of each individual rating is calculated on the train_set. Then the test_set is used to predict the rating and compare with the actual rating. The final table results show the probabilities of the edx, train_set, monte carlo using train_set, test_set and RMSE using the test_set.
  
prop_edx_ratings<- (prop.table(table(edx$rating)))                # Proportion of each edx ratings 
prop_train_ratings<- (prop.table(table(train_set$rating)))        # Proportion of each train_set ratings
prop_test_ratings<- (prop.table(table(test_set$rating)))          # Proportion of each test_set ratings

set.seed(1, sample.kind = "Rounding")

# Create the probability of each rating - seq (min,max,interval)
p <- function(x, y) mean(y == x)
rating <- seq(0.5,5,0.5)                                              

# Estimate the probability of each rating with Monte Carlo simulation
B <- 10000
M <- replicate(B, {
  s <- sample(train_set$rating, 100, replace = TRUE)
  sapply(rating, p, y= s)
})
prob <- sapply(1:nrow(M), function(x) mean(M[x,]))                        # Proportion of each train_set ratings with Monte Carlo simulation
prob

# Predict random ratings
y_hat_random <- sample(rating, size = nrow(test_set), 
                       replace = TRUE, prob = prob)

RMSE_rating<- RMSE(test_set$rating, y_hat_random)
RMSE_rating

#calculation of the RMSE for each rating. It is the same as sqrt(mean((true_ratings - predicted_ratings)^2))
individual_RMSE_rating <- data.frame (c(RMSE(test_set$rating==0.5, y_hat_random==0.5),         # Proportion of each test_set ratings
                                       RMSE(test_set$rating==1, y_hat_random==1),
                                       RMSE(test_set$rating==1.5, y_hat_random==1.5),
                                       RMSE(test_set$rating==2, y_hat_random==2),
                                       RMSE(test_set$rating==2.5, y_hat_random==2.5),
                                       RMSE(test_set$rating==3, y_hat_random==3),
                                       RMSE(test_set$rating==3.5, y_hat_random==3.5),
                                       RMSE(test_set$rating==4, y_hat_random==4),
                                       RMSE(test_set$rating==4.5, y_hat_random==4.5),
                                       RMSE(test_set$rating==5, y_hat_random==5)))                # validated by  sqrt(mean(((test_set$rating==5) - (y_hat_random==5))^2))




### create each rating proportion & RMSE final table
rating_prop<-data.frame (rating= c(0.5,1,1.5,2,2.5,3,3.5,4,4.5,5),prop_edx_ratings, prop_train_ratings, prob, prop_test_ratings, individual_RMSE_rating) %>%  # create final table
  select (rating, Freq, Freq.1, prob, Freq.2, c.RMSE.test_set.rating....0.5..y_hat_random....0.5...RMSE.test_set.rating.... ) 
  names(rating_prop)<-c("rating","probabilities edx", "probabilities train_set", "probabilities monte carlo train_set", "probabilities test_set", "RMSE test_set")


kable(rating_prop) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(6,bold =T ,color = "white" , background ="#D7261E")

#Results of the RMSE test_set for individual star rating using the Random prediction model were below the RMSE target.

#The RMSE of random prediction for the overall rating was very high (1.499), above the RMSE Capstone Target of 0.8649. RMSE of each individual rating met the RMSE Capstone Target.
### create RMSE for rating 
rating_RMSE_1 <- data.frame (Method = c("RMSE Target", " ", "Random prediction"), RMSE = c(0.8649," ",RMSE_rating))
#rating_RMSE_1

kable(rating_RMSE_1) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  row_spec(1,bold = T , color = "blue" ) %>%
  column_spec(2,bold =T) %>%
  row_spec(2,bold =T ,color = "white" , background ="grey") %>%
 row_spec(3,bold =T ,color = "white" , background ="#D7261E")

#### 2.3.2 Linear Models #### 

#P-value of the F-statistic is < 2.2e-16, which is highly significant y=mx+b. The only predictor that was significantly related to the rating was single_genresIMAX. All the other genres had Pr(>|t|) > 0.05. This means that changes in single_genresIMAX ratings are significantly associated to changes in ratings, while changes in all remaining individual single genres seem not to be significantly associated with ratings.
#single_genresIMAX coefficient suggests that for every 1 rating increase, it can be  expected a decrease of -1.3214286 *1 = 1.32  star units, on average. Meaning the maximum score would be 5 stars - 1.32 = 3.68 stars (since the slope is negative, the decrease ratio is 1.32 per unit)
#R-squared represents the correlation coefficient. Strong correlation means that R-squared is close to 1, positively or negatively.  In the rating ~ single_genres linear regression model, the multiple R-squared was 0.0221, demonstrating that there was no correlation. Only 2.21% of the variance in the measure of rating can be predicted by single genres.
#Residual standard error (RSE) is the measure of error of prediction. The lower the RSE is, the more accurate is the model.

set.seed(1, sample.kind = "Rounding")
fit_genres<-lm(rating ~ single_genres, data = edx)
fit_genres
summary(fit_genres)

#confidence intervals
confint(fit_genres)

#**rating ~ movie_rated_era + premier_year_era**
#There is no stronger correlation between Release decades (1980s, 1900s and 2000s) and related average ratings, although p-value was significant. P-value of the F-statistic is < 2.2e-16, which is also highly significant. The only predictors that were not significantly related to the rating were release decades 1980s, 1990s and 2000s. All the other decades prior 1980s had highly significant association with ratings.
#The coefficients suggesting a decrease in units were for the release decade 1990s and movies rated during 2000s (with-0.0587 and -0.1184530 star units respectively, on average). As a result, the maximum score would be 4.95 and 4.89 stars respectively. In the rating ~ movie_rated_era + premier_year_era Multiple linear regression (MLR), the multiple R-squared was 0.01827, demonstrating that there was also no correlation. Only 1.8% of the variance in the measurement of rating can be predicted by release decade and rating during 2000.

set.seed(1, sample.kind = "Rounding")
fit_decade<-lm(rating ~ movie_rated_era + premier_year_era, data = edx)
fit_decade
summary(fit_decade)

#confidence intervals
confint(fit_decade)


#Prediction 1 - Only mean of the ratings
#The initial prediction is test that all users will give the same rating to all movies and each rating is randomly distributed.
#formula is y_hat=mu+error 
mu <- mean(train_set$rating)             # Mean of observed values
RMSE_mu <- RMSE(test_set$rating, mu)
RMSE_mu


v_RMSE_mu <- RMSE(validation$rating, mu)
v_RMSE_mu

#Prediction 2 - Include Movie Effect (bi)
# Movie variability may be related to the fact that different movies will have different rating distribution. 
# Some maybe bias by the producer, cast, topic  this movie bias is expressed as bi in this formula: y_hat=mu+bi+error  (mean of the difference between the observed rating y and the mean μ).

# Movie effects (bi)
bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
head(bi)

#bi distribution 
bi %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("Movie Effect Distribution (bi) ") +
  theme_minimal()+ theme(panel.grid = element_blank(),axis.title.y = element_blank())
  
# Predict the rating with mean + bi with test_set
y_hat_bi <- mu + test_set %>% 
  left_join(bi, by = "movieId") %>% 
  .$b_i
RMSE_bi<- RMSE(test_set$rating, y_hat_bi)
RMSE_bi

# Predict the rating with mean + bi with validation
v_y_hat_bi <- mu + validation %>% 
  left_join(bi, by = "movieId") %>% 
  .$b_i
v_RMSE_bi<- RMSE(validation$rating, v_y_hat_bi)
v_RMSE_bi


#Prediction 3 - Include User Effect (bu)
# Different users have different rating distribution, this can also be biased by their own movie preference, physiological effect when they were wating the movie and rating the movies.
#Users can also be influence by the moving rating itself. The formula is y_hat=mu+bi+bu+error  

# User effect (bu)
bu <- train_set %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#User effect distribution shows to be more normally distributed
bu %>% ggplot(aes(x = b_u)) + 
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("User Effect Distribution (bu)") +
  theme_minimal()+ theme(panel.grid = element_blank(),axis.title.y = element_blank())

# Predict the rating with mean + bi+ bu with test_set
y_hat_bi_bu <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
RMSE_bi_bu <- RMSE(test_set$rating, y_hat_bi_bu)
RMSE_bi_bu


# Predict the rating with mean + bi+ bu with validation
v_y_hat_bi_bu <- validation %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
v_RMSE_bi_bu <- RMSE(validation$rating, v_y_hat_bi_bu)
v_RMSE_bi_bu

#Prediction 4 - Time Effect (t) - year rated
# Time can also be biased by the season. For example during valentine's day and fall season people may watch more romance movies, whereas during Halloween more horror movies.
#Also people can be influenced by critical events that occurred during a given period of time. The formula is *y_hat=mu+bi+bu+error*   

# Time effect (t)
t <- train_set %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%               #Timestamp in week
  left_join(bi, by = 'movieId') %>%
  left_join(bu, by = 'userId') %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))
t

#User effect distribution shows to be more normally distributed
t %>% ggplot(aes(x = b_t)) + 
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("Time Effect Distribution (bt) based on rating year") +
  theme_minimal()+ theme(panel.grid = element_blank(),axis.title.y = element_blank())

# Predict the rating with mean + bi+ bu = bt with test_set
y_hat_bi_bu_bt <- test_set %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  left_join(t, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred

RMSE_bi_bu_bt <- RMSE(test_set$rating, y_hat_bi_bu_bt)
RMSE_bi_bu_bt


# Predict the rating with mean + bi+ bu = bt with validation
v_y_hat_bi_bu_bt <- validation %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  left_join(t, by='date') %>%
  mutate(pred = mu + b_i + b_u +b_t) %>%
  .$pred

v_RMSE_bi_bu_bt <- RMSE(validation$rating, v_y_hat_bi_bu_bt)
v_RMSE_bi_bu_bt


#The time effect seems not make much difference at the RMSE validation results. It went from RMSE of 0.86585(adding movie and user effect) to  RMSE of 0.86581 (adding movie + user + time effect).

### create RMSE comparison table 

rating_RMSE2 <- data.frame (Method = c("RMSE Target"," ", "Random prediction", "Only mean of the ratings", "Adding Movie effects (bi)", "Adding Movie + User Effect (bu)", "Adding Movie + User + time Effect (bt)"), RMSE_test_set= c(0.8649," ", RMSE_rating,RMSE_mu,RMSE_bi, RMSE_bi_bu, RMSE_bi_bu_bt), RMSE_validation= c(0.8649," " , " ", v_RMSE_mu,v_RMSE_bi, v_RMSE_bi_bu, v_RMSE_bi_bu_bt))
#rating_RMSE2

kable(rating_RMSE2) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  row_spec(1,bold =T ,color = "blue") %>%
  row_spec(2,bold =T ,color = "white" , background ="gray") %>%
  row_spec(7,bold =T ,color = "white" , background ="#D7261E")

############################################################################################
# Note: code will take about 40min to run the matrix factorization - please be patience
############################################################################################

#### 2.3.3 matrix factorization #### 

#Trying to achieve the RMSE Target < 0.8649 and tired to fight the memory limit issues due to the large dataset. I decide to try the recosystem package to create matrix factorization.
# According to the package documentation <https://cran.r-project.org/web/packages/recosystem/recosystem.pdf> the default parameters are:
  #costp_l2 Tuning parameter, the L2 regularization cost for user factors. Can be specified as a numeric vector, with default value c(0.01,0.1).
  #costq_l2 Tuning parameter, the L2 regularization cost for item factors. Can be specified as a numeric vector, with default value c(0.01,0.1).
  #lrate Tuning parameter, the learning rate, which can be thought of as the step size in gradient descent. Can be specified as a numeric vector, with default value c(0.01,0.1).
  #niter Integer, the number of iterations. Default is 20.
  #nthread Integer, the number of threads for parallel computing. Default is 1.



set.seed(123, sample.kind = "Rounding") # This is a randomized algorithm

# Convert the train and test sets into recosystem input format
train_data <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId,
                                           rating     = rating,
                                           date     = date))
test_data  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating,
                                           date     = date))


validation_data  <-  with(validation,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating,
                                           date     = date))

# Create the model object
r <-  recosystem::Reco()

# Select the best tuning parameters. I used the parameters that people has been using wirh Reco() examples such as Qiu(2020)
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30),          # dim is number of factors 
                                       lrate = c(0.1, 0.2),          # learning rate
                                       costp_l2 = c(0.01, 0.1),      #regularization for P factors 
                                       costq_l2 = c(0.01, 0.1),      # regularization for  Q factors 
                                       nthread  = 4, niter = 10))    #convergence can be controlled by a number of iterations (niter) and learning rate (lrate)

# Train the algorithm  
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

# Calculate the predicted values using Reco test_data   
y_hat_reco <-  r$predict(test_data, out_memory())                    #out_memory(): Result should be returned as R objects
head(y_hat_reco, 10)

RMSE_reco <- RMSE(test_set$rating, y_hat_reco)
RMSE_reco

# Calculate the predicted values using Reco validation_data  
v_y_hat_reco <-  r$predict(validation_data, out_memory())                    #out_memory(): Result should be returned as R objects
head(v_y_hat_reco, 10)

v_RMSE_reco <- RMSE(validation$rating, v_y_hat_reco)
v_RMSE_reco

#### 3. Results

#This the final RMSE comparison table. It contains all the models used to predict ratings.
#As can been seen from the comparison table, the target RMSE of 0.8649 was almost reached after the Regularization qith Movie, user and time effect (RMSE of 0.8658). The best results of the RMSE were achieved when matrix factorization was implemented **0.78581** (around 9% below the RMSE target of 0.8649). 

### create RMSE comparison table 

rating_RMSE3 <- data.frame (Method = c("RMSE Target"," ", "Random prediction", "Only mean of the ratings", "Adding Movie effects (bi)", "Adding Movie + User Effect (bu)", "Adding Movie + User + time Effect (bt)", "Matrix Factorization (recosystem)"), RMSE_test_set= c(0.8649," ", RMSE_rating,RMSE_mu,RMSE_bi, RMSE_bi_bu, RMSE_bi_bu_bt, RMSE_reco), RMSE_validation= c(0.8649," " , " ", v_RMSE_mu,v_RMSE_bi, v_RMSE_bi_bu, v_RMSE_bi_bu_bt,v_RMSE_reco))
#rating_RMSE3

kable(rating_RMSE3) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ) %>%
  row_spec(1,bold =T ,color = "blue") %>%
  row_spec(2,bold =T ,color = "white" , background ="gray") %>%
  row_spec(8,bold =T ,color = "white" , background ="#D7261E")


#### 4. Conclusion #### 

#This project was a great wrap-up of the certification, it gave us freedom to build our own analysis while we all had in mind a common target. It showcased the importance of validate the data transformation as well gave me different perspectives of data insights.
#After running my recommender system algorithms using random prediction, Linear model with regularized effects on movies +users + time effect I reached the *RMSE of* **0.86585** *using the validation* and **0.8646** *using the test_set* which was below the  Capstone RMSE Target of *0.8649*. Since I was not fully satisfied with my RMSE results, I decided to try Matrix Factorization method (recosystem), as some of the discussion highly recommended. I was very impressed to reach RMSE of **0.78581** *using the validation* (9% less than the Capstone RMSE Target).
#Limitations were mostly related with the dataset size and the memory limit issues that I encounter over several chunks of codes. After some research, I used gc to clean unused memory and I was increased the memory limit. Errors disappears, but the time to run the codes were in between 1h40min and 2h30min. This means that I spent probably 60% of my time waiting for the code to run, instead of been developing/improving my codes.
#As future work, I would be curious to see how smaller datasets and different combinations of train_set and test_set would impact the RMSE results.


#### 5.Reference ####

#Perrier, A. Effective Amazon Machine Learning . Published by Packt Publishing, 2017.

#Forte, R and Miller, J. Mastering Predictive Analytics with R - Second Edition. Published by Packt Publishing, 2017.

#Qiu, Y. recosystem: Recommender System Using Parallel Matrix Factorization https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html, 2020.
