# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:42:30 2018

@author: Nitesh Choudhary (https://www.linkedin.com/in/nitesh-choudhary/)
"""

# Months of 2017-2018 season
months=['october','november','december','january','february','march','april']

# months for predicted matches
month=['february','march','april'] 

# Importing Libraries
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Getting team stats
def team_row(team_code):
    team_df=pd.DataFrame()
    team_url='https://www.basketball-reference.com/teams/'+ team_code +'/'
    team_html=urlopen(team_url)
    team_soup=BeautifulSoup(team_html)
    team_headers = [th.getText() for th in team_soup.findAll('tr', limit=2)[0].findAll('th')]
    team_row = [[td.getText() for td in team_soup.findAll('tr', limit=2)[1].findAll('td')]]
    team_df = team_df.append(pd.DataFrame(team_row, columns=team_headers[1:]),ignore_index=True)
    return team_df

# Getting Matches 
def get_matches(month):
    matches_url='https://www.basketball-reference.com/leagues/NBA_2018_games-'+ month +'.html'
    html = urlopen(matches_url)
    soup = BeautifulSoup(html)
    column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    data_rows = soup.findAll('tr')[1:]
    matches_data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]
    df = pd.DataFrame(matches_data, columns=column_headers[1:])
    df=df[['Visitor/Neutral','Home/Neutral','PTS']]
    df.columns=['Visitor/Net','Home/Net','Visitor_Points','Home_Points']
    df['Home_Win']=np.where(pd.to_numeric(df['Home_Points'])>pd.to_numeric(df['Visitor_Points']),1,0)
    df['Home_Win']=np.where(df['Home_Points']=="","",df['Home_Win'])
    return df

# Getting Schedule
def get_date_time(month):    
    matches_url='https://www.basketball-reference.com/leagues/NBA_2018_games-'+ month +'.html'
    html = urlopen(matches_url)
    soup = BeautifulSoup(html)
    column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    data_rows = soup.findAll('tr')[1:]
    matches_date = [[th.getText() for th in data_rows[i].findAll('th')] for i in range(len(data_rows))]
    matches_time = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]
    matches_time = pd.DataFrame(matches_time, columns=column_headers[1:])
    x=pd.concat([pd.DataFrame(matches_date,columns=['Date']),matches_time],axis=1)
    x=x[x.iloc[:,3]==''][['Date','Start (ET)']].reset_index()
    return x[['Date','Start (ET)']]

# Fetching data
team_east=['TOR','BOS','CLE','WAS','IND','MIL','PHI','MIA','DET','CHA','NYK','CHI','NJN','ORL','ATL']
team_west=['HOU','GSW','SAS','MIN','OKC','DEN','POR','NOH','LAC','UTA','LAL','MEM','SAC','DAL','PHO']
team_codes=team_east+team_west
team_stats=pd.DataFrame()
for i in team_codes:
    team_stats=team_stats.append(team_row(i),ignore_index=True)
    
team_matches=pd.DataFrame()
for k in months:
    team_matches=team_matches.append(get_matches(k),ignore_index=True)

# Preparing data    
vn_team_stats=team_stats.copy()
vn_team_stats.columns=['VN_' + str(col) for col in vn_team_stats.columns]
VN_team_stats=vn_team_stats[['VN_Team','VN_W/L%', 'VN_Finish', 'VN_SRS','VN_Pace', 'VN_Rel_Pace', 'VN_ORtg', 'VN_Rel_ORtg', 'VN_DRtg','VN_Rel_DRtg']]
team_matches=team_matches.merge(VN_team_stats,how='left',left_on='Visitor/Net',right_on='VN_Team')


hn_team_stats=team_stats.copy()
hn_team_stats.columns=['HN_' + str(col) for col in hn_team_stats.columns]
HN_team_stats=hn_team_stats[['HN_Team','HN_W/L%', 'HN_Finish', 'HN_SRS','HN_Pace', 'HN_Rel_Pace', 'HN_ORtg', 'HN_Rel_ORtg', 'HN_DRtg','HN_Rel_DRtg']]
team_matches=team_matches.merge(HN_team_stats,how='left',left_on='Home/Net',right_on='HN_Team')


team_matches=team_matches[['Visitor/Net', 'Home/Net', 'Visitor_Points', 'Home_Points',
       'VN_W/L%', 'VN_Finish', 'VN_SRS', 'VN_Pace', 'VN_Rel_Pace',
       'VN_ORtg', 'VN_Rel_ORtg', 'VN_DRtg', 'VN_Rel_DRtg','HN_W/L%', 'HN_Finish', 
       'HN_SRS', 'HN_Pace', 'HN_Rel_Pace', 'HN_ORtg',
       'HN_Rel_ORtg', 'HN_DRtg', 'HN_Rel_DRtg', 'Home_Win']]

# Training
x_train=team_matches[team_matches['Home_Points']!=''].iloc[:,4:-1]
y_train=team_matches[team_matches['Home_Points']!=''].iloc[:,-1]

x_train=x_train.apply(pd.to_numeric)
y_train=y_train.apply(pd.to_numeric)

# Unknown/Prediction dataset
x_unknown=team_matches[team_matches['Home_Points']==''].iloc[:,4:-1]
x_unknown=x_unknown.apply(pd.to_numeric)

# Visuals - Pairplot
sns.pairplot(team_stats.iloc[:,2:-3].apply(pd.to_numeric))

# Visuals - Correlation Heatmap
a4_dims = (10, 7)
fig, ax = plt.subplots(figsize=a4_dims)
hm=team_stats.iloc[:,4:-3].apply(pd.to_numeric).corr()
sns.heatmap(ax=ax,data=hm)

# Modeling
glm= linear_model.LogisticRegression(C=8)
glm.fit(x_train,y_train)

# Beta Coefficients
pd.concat([pd.Series(x_train.columns,name='Attributes'),pd.Series(glm.coef_[0],name='Beta Log Coeff')],axis=1)

# Model Evaluation
y_pred=glm.predict(x_train)
y_proba=glm.predict_proba(x_train)

# Confusion Matrix
pd.crosstab(y_train, y_pred, rownames=['True'], colnames=['Predicted'], margins=False)

# AUC calculation
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(y_train, y_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])    


# Predictions
y_hat=pd.DataFrame(glm.predict_proba(x_unknown))
y_hat.columns=['Visitor_team_prob','Home_team_prob']
x_hat=team_matches[team_matches['Home_Points']==''].iloc[:,0:2].reset_index()
upcoming_matches=(pd.concat([x_hat,y_hat],axis=1)).drop(['index'],axis=1)

# Formatting (Adding date and time)
match_pred=pd.DataFrame()
for m in month:
    match_pred=match_pred.append(get_date_time(m),ignore_index=True)

# Final Result
upcoming_matches=pd.concat([match_pred,upcoming_matches],axis=1)
upcoming_matches.to_csv('nba_preds.csv',sep=',')
