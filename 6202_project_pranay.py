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
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib import rcParams

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
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
actions=list(data1['action_type'].unique())
shots_percent=[]
total_size=data1['shot_made_flag'].size

for i in actions:
    score_no=data_yes.loc[data_yes['action_type']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['action_type']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent[:10],columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of top 10 action_type')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
plt.savefig('Graphs/2.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# 3. From which shot_zone_area did he score/fail the most
areas=list(data1['shot_zone_area'].unique())
shots_percent=[]

for i in areas:
    score_no=data_yes.loc[data_yes['shot_zone_area']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['shot_zone_area']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of shot_zone_area')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
plt.savefig('Graphs/3.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 4. From which shot_zone_basic did he score/fail the most
basics=list(data1['shot_zone_basic'].unique())
shots_percent=[]
for i in basics:
    score_no=data_yes.loc[data_yes['shot_zone_basic']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['shot_zone_basic']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of shot_zone_basic')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
plt.savefig('Graphs/4.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 5. From which shot_zone_range did he score/fail the most
ranges=list(data1['shot_zone_range'].unique())
shots_percent=[]
for i in ranges:
    score_no=data_yes.loc[data_yes['shot_zone_range']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['shot_zone_range']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of shot_zone_range')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
plt.savefig('Graphs/5.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 6. Against which team did he best/worst perform
opponent_teams=data1['opponent'].unique()
opp_points={}

for i in opponent_teams:
    points=data1.loc[data1['opponent']==i,'shot_made_flag'] * data1.loc[data1['opponent']==i,'shot_type']
    matches=data1.loc[data1['opponent']==i,'game_id'].unique().size
    opp_points[points.sum()/matches]=i

opp_sum1=list(opp_points.keys())
opp_sum1.sort(reverse=True)
opp_sum_teams1=[]
for i in opp_sum1[:5]:
    opp_sum_teams1.append(opp_points[i])
opp_sum_teams2=[]
for i in opp_sum1[-1:-6:-1]:
    opp_sum_teams2.append(opp_points[i])

plt.figure(figsize=(10,8))
plots=sns.barplot(opp_sum_teams1,opp_sum1[:5])
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.title('barplot of 5 avg score vs opponent teams - that kobe averaged highest')
plt.savefig('Graphs/6_1.jpeg', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,8))
plots=sns.barplot(opp_sum_teams2,opp_sum1[-1:-6:-1])
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.title('barplot of 5 avg score vs opponent teams - that kobe averaged least')
plt.savefig('Graphs/6_2.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# 7. accuracy per season accuracy = yes/yes+no
seasons=list(data1['season'].unique())
seasons.sort()
seasons_dict={}
for i in seasons:
    accuracy=(data_yes.loc[data_yes['season']==i,'shot_made_flag'].size)/(data1.loc[data1['season']==i,'shot_made_flag'].size)
    seasons_dict[i]=accuracy

plt.figure(figsize=(10,8))
plt.plot(list(seasons_dict.keys()),list(seasons_dict.values()))
plt.scatter(list(seasons_dict.keys()),list(seasons_dict.values()),s=40)
plt.title('accuracy vs seasons')
plt.xticks(rotation=90)
plt.xlabel('seasons')
plt.ylabel('accuracy')
plt.tight_layout()
plt.savefig('Graphs/7.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 8. shots vs period
periods=list(data1['period'].unique())
shots_percent=[]
for i in periods:
    score_no=data_yes.loc[data_yes['period']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['period']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('shots attempted vs period')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
plt.savefig('Graphs/8.jpeg', dpi=300, bbox_inches='tight')
plt.show()
