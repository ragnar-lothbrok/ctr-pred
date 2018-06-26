import pandas
import numpy as np

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

teams = ['Russia','Belgium','Germany','England','Spain','Poland','Iceland','Serbia','France','Portugal','Switzerland','Croatia','Sweden','Denmark','Iran','South Korea','Japan','Saudi Arabia','Australia','Mexico','Costa Rica','Panama','Brazil','Uruguay','Argentina','Colombia','Peru']


team_players = pandas.read_csv("04 2017 FIFA Players Database.csv")
for team in teams:
    # print team
    one_team = team_players[team_players['Nationality'] == team]
    # print one_team['Rating'].mean()


#This piece of code is caluclating all matches which all happended in history

history_matches =  pandas.read_csv("02 Historical Data on World Cup Matches.csv")
history_matches = history_matches.loc[history_matches['Year'] > 2000]
history_matches = history_matches[['Stadium','Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name']]

history_matches['diff'] = 0
history_matches['result'] = ""

print len(history_matches.index)
history_matches = history_matches.loc[history_matches['Home Team Name'].isin(teams)]
print len(history_matches.index)
print history_matches['Home Team Name'].unique()


for index, row in history_matches.iterrows():
    row["diff"] = long(row['Home Team Goals']) - long(row['Away Team Goals'])
    if long(row['Home Team Goals']) > long(row['Away Team Goals']):
        history_matches['result'] = "W"
    elif long(row['Home Team Goals']) < long(row['Away Team Goals']):
        history_matches['result'] = "L"
    else:
        history_matches['result'] = "D"

print "Matches played before world cup \t"
print len(history_matches.index)

#This piece of code is caluclating all matches which has happended this year and which are yet yo be played

current_matches =  pandas.read_csv("05 2018 World Cup Match Fixtures.csv")
current_matches = current_matches.loc[current_matches['Home Team Name'].isin(teams)]
current_matches = current_matches[['Stadium', 'Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name']]
current_matches['diff'] = 0
current_matches['result'] = ""

print "Before filtering all matches \t"
print len(current_matches.index)

matches_to_be_played = current_matches[np.isfinite(current_matches['Home Team Goals']) == False]

current_matches = current_matches[np.isfinite(current_matches['Home Team Goals'])]

for index, row in current_matches.iterrows():
    row["diff"] = long(row['Home Team Goals']) - long(row['Away Team Goals'])
    if long(row['Home Team Goals']) > long(row['Away Team Goals']):
        current_matches['result'] = "W"
    elif long(row['Home Team Goals']) < long(row['Away Team Goals']):
        current_matches['result'] = "L"
    else:
        current_matches['result'] = "D"

print "After filtering all played matches \t"
print len(current_matches.index)

print "Matches to be played \t"
print len(matches_to_be_played.index)

print list(current_matches)

#combining both data frames
all_macthes_playes = pandas.concat([current_matches, history_matches], ignore_index=True)
print "Total Matches played \t"
print len(all_macthes_playes.index)

#This will fetch few more columns on the basis of team
team_stats =  pandas.read_csv("01 2018 World Cup Team Statistics.csv")
team_stats = team_stats.loc[team_stats['team'].isin(teams)]
filtered_team_stats = team_stats[['team', 'Offence_rating', 'Defence_rating', 'Average_age', 'Average_height', 'all_time_fifa_ranking', 'total_world_cup_points', 'total_worldcup_appearances']]
print list(filtered_team_stats)

#data set for the matches already played
filtered_team_stats = filtered_team_stats.rename(columns={'team': 'Home Team Name'})
merged_matches = all_macthes_playes.set_index('Home Team Name').join(filtered_team_stats.set_index('Home Team Name'))
merged_matches = merged_matches.drop(columns=['Away Team Goals','Home Team Goals','diff'])
print list(merged_matches)
merged_matches.to_csv('train_dataset.csv', sep=',', encoding='utf-8')

#data set for the matches to be played
merged_unplayed_macthes = matches_to_be_played.set_index('Home Team Name').join(filtered_team_stats.set_index('Home Team Name'))
merged_unplayed_macthes = merged_unplayed_macthes.drop(columns=['Away Team Goals','Home Team Goals','diff'])
print list(merged_unplayed_macthes)
merged_unplayed_macthes.to_csv('test_dataset.csv', sep=',', encoding='utf-8')

#https://www.kaggle.com/cbrogan/xgboost-example-python


