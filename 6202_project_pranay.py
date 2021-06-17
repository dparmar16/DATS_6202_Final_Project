import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'

data=pd.read_csv('data.csv')

# printing the dataset shape
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])

# printing the dataset observations
print("Dataset first few rows:\n ")
print(data.head(6))

# printing the struture of the dataset
print("Dataset info:\n ")
data.info()

#drop missing values
data1=data.dropna()

#drop columns
data1.drop(['game_event_id','lat','loc_x','loc_y','lon','team_id','team_name','game_date','matchup','shot_id'],axis=1,inplace=True)

data1.loc[ data1['shot_zone_area']=='Right Side(R)','shot_zone_area' ] = 0
data1.loc[data1['shot_zone_area']=='Right Side Center(RC)','shot_zone_area']=1
data1.loc[data1['shot_zone_area']=='Center(C)','shot_zone_area']=2
data1.loc[data1['shot_zone_area']=='Left Side Center(LC)','shot_zone_area']=3
data1.loc[data1['shot_zone_area']=='Left Side(L)','shot_zone_area']=4
data1.loc[data1['shot_zone_area']=='Back Court(BC)','shot_zone_area']=5

data1.loc[data1['shot_type']=='2PT Field Goal','shot_type']=2
data1.loc[data1['shot_type']=='3PT Field Goal','shot_type']=3

# EDA analysis

#1. average points per game - only 2FG and 3FG included
points_scored=data1.shot_made_flag * data1.shot_type

games_no=data1['game_id'].unique().size

PPG=points_scored.sum()/games_no

# scored data
data_yes=data1[data1['shot_made_flag']==1]
# failed to score data
data_no=data1[data1['shot_made_flag']==0]

# 2. Which action_type worked/failed the most
plt.figure(figsize=(10,8))
data1['action_type'].value_counts()[:10,].plot(kind='bar',)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_yes['action_type'].value_counts()[:10,].plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_no['action_type'].value_counts()[:10,].plot(kind='bar')
plt.tight_layout()
plt.show()



# 3. From which shot_zone_area did he score/fail the most
plt.figure(figsize=(10,8))
data1['shot_zone_area'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_yes['shot_zone_area'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_no['shot_zone_area'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()


# 4. From which shot_zone_basic did he score/fail the most
plt.figure(figsize=(10,8))
data1['shot_zone_basic'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_yes['shot_zone_basic'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_no['shot_zone_basic'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

# 5. From which shot_zone_range did he score/fail the most
plt.figure(figsize=(10,8))
data1['shot_zone_range'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_yes['shot_zone_range'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
data_no['shot_zone_range'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

opponent_teams=data1['opponent'].unique()
opp_points={}

for i in opponent_teams:
    points=data1.loc[data1['opponent']==i,'shot_made_flag'] * data1.loc[data1['opponent']==i,'shot_type']
    opp_points[points.sum()]=i

opp_sum1=list(opp_points.keys())
opp_sum1.sort(reverse=True)
opp_sum_teams1=[]
for i in opp_sum1[:5]:
    opp_sum_teams1.append(opp_points[i])
opp_sum_teams2=[]
for i in opp_sum1[-1:-6:-1]:
    opp_sum_teams2.append(opp_points[i])

plt.bar(opp_sum_teams1,opp_sum1[:5])
plt.show()

plt.bar(opp_sum_teams2,opp_sum1[-1:-6:-1])
plt.show()