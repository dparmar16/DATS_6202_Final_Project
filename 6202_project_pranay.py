import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
# data1.drop(['game_event_id','lat','loc_x','loc_y','lon','team_id','team_name','game_date','matchup','shot_id'],axis=1,inplace=True)

# data1.loc[ data1['shot_zone_area']=='Right Side(R)','shot_zone_area' ] = 0
# data1.loc[data1['shot_zone_area']=='Right Side Center(RC)','shot_zone_area']=1
# data1.loc[data1['shot_zone_area']=='Center(C)','shot_zone_area']=2
# data1.loc[data1['shot_zone_area']=='Left Side Center(LC)','shot_zone_area']=3
# data1.loc[data1['shot_zone_area']=='Left Side(L)','shot_zone_area']=4
# data1.loc[data1['shot_zone_area']=='Back Court(BC)','shot_zone_area']=5

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

# 6. Against which team did he best/worst perform
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
for i in opp_sum1[-1:-7:-1]:
    opp_sum_teams2.append(opp_points[i])

plt.bar(opp_sum_teams1,opp_sum1[:5])
plt.show()

plt.bar(opp_sum_teams2,opp_sum1[-1:-7:-1])
plt.show()


# 7. accuracy per season accuracy = yes/yes+no
seasons=list(data1['season'].unique())
seasons.sort()
seasons_dict={}
for i in seasons:
    accuracy=(data_yes.loc[data_yes['season']==i,'shot_made_flag'].size)/(data1.loc[data1['season']==i,'shot_made_flag'].size)
    seasons_dict[i]=accuracy

plt.plot(list(seasons_dict.keys()),list(seasons_dict.values()))
plt.scatter(list(seasons_dict.keys()),list(seasons_dict.values()),s=40)
plt.title('accuracy vs seasons')
plt.xticks(rotation=90)
plt.xlabel('seasons')
plt.ylabel('accuracy')
plt.tight_layout()
plt.show()


# 8. shots vs period
periods=list(data1['period'].unique())
periods_dict={}
total_size=data1['shot_made_flag'].size
shots_percent1=[]

for i in periods:
    score_no=data1.loc[data1['period']==i,'shot_made_flag'].size
    periods_dict[i]=score_no
    shots_percent1.append((score_no*100)/total_size)

plt.figure(figsize=(8, 8))
plots = sns.barplot(x=list(periods_dict.keys()), y=list(periods_dict.values()))
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent1[z]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

plt.title('shots attempted per period')
plt.xlabel('period/quarters')
plt.ylabel('number of shots')
plt.show()

periods_dict_yes={}
shots_percent2=[]
for i in periods:
    score_no=data_yes.loc[data_yes['period']==i,'shot_made_flag'].size
    periods_dict_yes[i]=score_no
    shots_percent2.append((score_no*100)/total_size)

plt.figure(figsize=(8, 8))
plots = sns.barplot(x=list(periods_dict_yes.keys()), y=list(periods_dict_yes.values()))
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent2[z]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

plt.title('shots scored per period')
plt.xlabel('period/quarters')
plt.ylabel('number of shots')
plt.show()




