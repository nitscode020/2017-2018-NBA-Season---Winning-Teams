# 2017-2018-NBA-Season---Winning-Teams
Predicting the winning teams for 2017-2018 NBA season matches 

In this project, I have built a single-game win probability model. The project deliverable is a method which predicts the probability 
of the home team winning the game using only data available before tipoff. 

I took data from online sources mostly websites using Python. I wrote different functions to fetch data for teams, matches and schedules. 
Then I used these functions to fetch data from different pages of websites.

I used Python 3.6 to complete the entire project. Libraries that I used –
•	Urllib for accessing webpages 
•	Bs4 for using beautiful soup to read webpage html
•	Pandas to provide structure to fetched data
•	Numpy for matrix calculations
•	Sklearn for modeling and evaluation
•	Matplotlib and Seaborn for visualizations

Data Collection Process
The data collection process is currently automated. To predict the matches for this season I have used the statistics for this season. 
Current code scraps the websites and collects the data. This data consists of required training data (team stats, previous matches, wins, 
losses, ratings, points etc.) and upcoming matches schedule. Training data is then analyzed and used to train a logistic regression model. 
Logistic model then predicts the probabilities of winning for home and visitor teams in upcoming matches using the schedule that is 
scraped from the website. 

The probabilities for the upcoming matches are exported in a csv file with name ‘nba_preds.csv’.


