'''
# Conduct EDA
# Identify features
# Do pre-processing
# Fit MLP and RF
# Compare output
'''

# ----------------------------------------
### Data import ###
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib import rcParams

# Look at dataset
data = pd.read_csv('data.csv', header=0)
print(data.head)
print(data.columns)
'''
['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat',
       'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs',
       'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag',
       'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
       'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id',
       'made']
'''

# Get column datatypes and summaries
print(data.dtypes)

# Drop rows which were intended for Kaggle competition
# These rows are missing information about whether the shot was made or not
#
data1 = data.dropna()
#data1 = data1[(data1['shot_made_flag'] == 1) | (data1['shot_made_flag'] == 0)]
data1['made'] = np.where(data1['shot_made_flag'] == 1.0, 'Made', 'Missed')


# ----------------------------------------
### Pre-Processing ###


# Add avg field goal percentage by distance - do (1) bins and (2) continuous line
#pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r/r.sum(), axis=1)
#pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r.sum(), axis=1)
distances = pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r.sum(), axis=1).index
distance_attempts = pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r.sum(), axis=1)
distance_percent = pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r/r.sum(), axis=1)['Made']

fig, ax1 = plt.subplots()

color1 = 'tab:red'
ax1.set_xlabel('Distance from basket')
ax1.set_ylabel('Attempts', color=color1)
ax1.plot(distances, distance_attempts, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color2 = 'tab:blue'
ax2.set_ylabel('Field Goal Percentage', color=color2)  # we already handled the x-label with ax1
ax2.plot(distances, distance_percent, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# Features to consider
# ['action_type', 'combined_shot_type', 'period', 'playoffs',
#        'season', 'shot_distance', 'shot_made_flag',
#        'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
#        'made']

# Look for variation in field goal percentage in the feature values
# If a feature has that variation, we should add it to the model

# May or may not use action type since it has too many values, many with small sample
pd.crosstab(data1.action_type, data1.made)
pd.crosstab(data1.action_type, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will use combined shot type since it has more sample in each category and has variation
pd.crosstab(data1.combined_shot_type, data1.made)
pd.crosstab(data1.combined_shot_type, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will use period variable
pd.crosstab(data1.period, data1.made)
pd.crosstab(data1.period, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will may use the playoffs variable, but likelihood of making shot is nearly same in both classes
pd.crosstab(data1.playoffs, data1.made)
pd.crosstab(data1.playoffs, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will use season variable, although there is very little variation between seasons
pd.crosstab(data1.season, data1.made)
pd.crosstab(data1.season, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will use shot type, large variation between two pointers and three pointers
pd.crosstab(data1.shot_type, data1.made)
pd.crosstab(data1.shot_type, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will use shot zone area, percentages vary greatly between classes
pd.crosstab(data1.shot_zone_area, data1.made)
pd.crosstab(data1.shot_zone_area, data1.made).apply(lambda r: r/r.sum(), axis=1)

# We will use shot zone basic, it has variation between groups
pd.crosstab(data1.shot_zone_basic, data1.made)
pd.crosstab(data1.shot_zone_basic, data1.made).apply(lambda r: r/r.sum(), axis=1)

# will use shot zone range, it has variation between groups
pd.crosstab(data1.shot_zone_range, data1.made)
pd.crosstab(data1.shot_zone_range, data1.made).apply(lambda r: r/r.sum(), axis=1)

# Drop all columns that aren't going into the model
data1.drop(['game_event_id', 'game_id', 'lat',
            'loc_x', 'loc_y', 'lon', 'minutes_remaining',
            'seconds_remaining',
            'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id',
            'made'], axis=1, inplace=True)

# Change shot distance max to 40 to reduce variable range caused by outliers
data1['shot_distance'] = np.where(data1['shot_distance'] >= 40, 40, data1['shot_distance'])

# Replace some columns in anticipation of encoder
# Encoder works in alphabetical order
data1['combined_shot_type'].replace('Jump Shot', 'EE', inplace=True)
data1['combined_shot_type'].replace('Layup', 'BB', inplace=True)
data1['combined_shot_type'].replace('Dunk', 'AA', inplace=True)
data1['combined_shot_type'].replace('Tip Shot', 'CC', inplace=True)
data1['combined_shot_type'].replace('Hook Shot', 'DD', inplace=True)


data1['shot_zone_area'].replace('Center(C)', 'CC', inplace=True)
data1['shot_zone_area'].replace('Right Side Center(RC)', 'DD', inplace=True)
data1['shot_zone_area'].replace('Right Side(R)', 'EE', inplace=True)
data1['shot_zone_area'].replace('Left Side Center(LC)', 'BB', inplace=True)
data1['shot_zone_area'].replace('Left Side(L)', 'AA', inplace=True)

data1['shot_zone_basic'].replace('Restricted Area', 'AA', inplace=True)
data1['shot_zone_basic'].replace('In The Paint (Non-RA)', 'BB', inplace=True)
data1['shot_zone_basic'].replace('Mid-Range', 'CC', inplace=True)
data1['shot_zone_basic'].replace('Left Corner 3', 'DD', inplace=True)
data1['shot_zone_basic'].replace('Above the Break 3', 'EE', inplace=True)
data1['shot_zone_basic'].replace('Right Corner 3', 'FF', inplace=True)
data1['shot_zone_basic'].replace('Backcourt', 'GG', inplace=True)

data1['shot_zone_range'].replace('Less Than 8 ft.', 'AA', inplace=True)
data1['shot_zone_range'].replace('8-16 ft.', 'BB', inplace=True)
data1['shot_zone_range'].replace('16-24 ft.', 'CC', inplace=True)
data1['shot_zone_range'].replace('24+ ft.', 'DD', inplace=True)
data1['shot_zone_range'].replace('Back Court Shot', 'EE', inplace=True)


# Separate features and target
X = data1.drop(['shot_made_flag'], axis=1)
y = data1['shot_made_flag']

# Transform variables
# ['action_type', 'combined_shot_type', 'period', 'playoffs',
#        'season', 'shot_distance'
#        'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
#        ]
# Shot made flag needs to be label encoded
le = LabelEncoder()
mm = MinMaxScaler()

# May or may not use action type since it has too many values, many with small sample
# Decide to use it, as neural networks aren't as affected by curse of dimensionality
# Label encode it and scale 0 to 1
# Do this for the other columns as well
columns_to_transform =  ['action_type', 'combined_shot_type', 'period', 'playoffs',
                         'season', 'shot_distance'
                                   'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
                         ]

for c in X.columns:
    X[c] = le.fit_transform(X[c])
    # X[c] = mm.fit_transform(X[c].values.reshape(-1,1))


# Label encode y
y = le.fit_transform(y)
print(y)


# Correlation matrix of features to output
# TO DO

# ----------------------------------------
### Modeling ###
## Random Forest
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#%%-----------------------------------------------------------------------
#perform training with random forest with all columns
# specify random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
clf = RandomForestClassifier(n_estimators=10,random_state=100)

# perform training
clf.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
#plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data1.loc[:,data1.columns!='shot_made_flag'].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
# featuK=clf.feature_importances_.argsort()[::-1][:5]
featuK=f_importances.index[:6]
newX_train = X_train.loc[:,featuK]

# select the testing dataset on k-features
newX_test = X_test.loc[:, featuK]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=10,random_state=100)

# train the model
clf_k_features.fit(newX_train, y_train)


#%%-----------------------------------------------------------------------
#make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)



# %%-----------------------------------------------------------------------
# calculate metrics for all features

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# calculate metrics Using K features
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)

# %%-----------------------------------------------------------------------
# confusion matrix for all features
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data1['shot_made_flag'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.title('confusion matrix for all features')
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# confusion matrix for K features
conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = data1['shot_made_flag'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.title('confusion matrix for K features')
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# for this model - n_estimators=10, randomstate=100
# we can observe that the accuracy score and ROC_AUC increased slightly when we use feature reduction so we can build
# model with only 6 features - ['season', 'shot_distance', 'action_type', 'period', 'shot_zone_area','combined_shot_type']
# and from confusion matrix we can observe that true negatives increased  and true positives decreased very slightly.
# so the model with only six features is best model


# Train-test Split
X_train_NN, X_test_NN, y_train_NN, y_test_NN = train_test_split(X, y, random_state=100)
clf_NN = MLPClassifier(random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN = clf_NN.predict(X_test_NN)
y_pred_score_NN = clf_NN.predict_proba(X_test_NN)

print("\n")
print('MLP modelling')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN))
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test_NN,y_pred_score_NN[:,1]) * 100)

# confusion matrix for all features
conf_matrix = confusion_matrix(y_test_NN, y_pred_NN)
class_names = data1['shot_made_flag'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.title('confusion matrix for all features - MLP')
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
# featuK=clf.feature_importances_.argsort()[::-1][:5]
# featuK=f_importances.index[:6]
newX_train_NN = X_train_NN.loc[:,featuK]

# select the testing dataset on k-features
newX_test_NN = X_test_NN.loc[:, featuK]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify MLP classifier and train
clf_k_features_NN = MLPClassifier(random_state=100, max_iter=200).fit(newX_train_NN, y_train_NN)

#%%-----------------------------------------------------------------------

# prediction on test using k features
y_pred_k_features_NN = clf_k_features_NN.predict(newX_test_NN)
y_pred_k_features_score_NN = clf_k_features_NN.predict_proba(newX_test_NN)

# calculate metrics Using K features
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_k_features_NN))
print("\n")
print("Mean squared error : ", mean_squared_error(y_test_NN, y_pred_k_features_NN))
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test_NN,y_pred_k_features_score_NN[:,1]) * 100)


# confusion matrix for K features
conf_matrix = confusion_matrix(y_test_NN, y_pred_k_features_NN)
class_names = data1['shot_made_flag'].unique()



df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.title('confusion matrix for K features - MLP')
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# for this model MLP- random_state=100, max_iter=200
# we can observe that the accuracy score and ROC_AUC decreased slightly when we use feature reduction so we can build
# model with only 6 features - ['season', 'shot_distance', 'action_type', 'period', 'shot_zone_area','combined_shot_type']
# and from confusion matrix we can observe that true negatives increased  and true positives decreased.
# so the model with only six features is not best model - select the model with all the features
# though the MLP model has better accuracy,roc and high number of true negatives than the random forest, the true poitives
# have decreased from the random forest, since the main goal of this analysis is to predict the shots made, MLP model doesn't hold
# to random forest model