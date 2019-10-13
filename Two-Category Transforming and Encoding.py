#Load the modules
import pandas as pd
import numpy as np
import random

#Read the data file
data = pd.read_csv("C:\\Users\\Eric\\Desktop\\df_preprocessed.csv")
data = data.dropna()

#Delete the variables
delete_var_stage1 = ['Unnamed: 0','meeting_id','meeting_date','conditions',
'race_name','race_abbrev_name','race_type_id','scheduled_time','off_time',
'winning_time_disp','standard_time_disp','loaded_at.x','meeting_status',
'race_title','showcase','name','foaling_date','in_race_comment','trainer_name',
'owner_name','owner_id','jockey_name','dam_name','dam_id','sire_name',
'dam_sire_name','forecast_price','starting_price','distance_beaten',
'distance_behind_winner','last_race_type_id','jockey_colours']
data.drop(delete_var_stage1,axis=1, inplace=True)

#Classifier the attributes
race = ['race_id', 'course', 'race_type', 'race_num', 'going', 'direction',
        'class', 'draw_advantage', 'handicap', 'all_weather', 'seller',
        'claimer', 'apprentice', 'maiden', 'amateur', 'num_runners',
        'num_finishers', 'rating', 'min_age', 'distance_yards', 'added_money',
        'official_rating', 'speed_rating', 'private_handicap',
        'winning_time_secs', 'standard_time_secs', 'country', 'track_type',
        'advanced_going', 'trifecta', 'age_range', 'penalty_value',
        'prize_pos_1', 'prize_pos_2', 'prize_pos_3', 'prize_pos_4',
        'last_winner_year', 'last_winner_runners', 'last_winner_age',
        'last_winner_bred', 'last_winner_weight', 'last_winner_sp_decimal']

runner = ['runner_id', 'colour', 'distance_travelled', 'gender', 'age', 'bred',
            'cloth_number', 'official_rating.1', 'official_rating_type',
            'speed_rating_type', 'private_handicap_type', 'trainer_id', 'jockey_id',
            'sire_id', 'dam_sire_id', 'forecast_price_decimal',
            'starting_price_decimal', 'position_in_betting',
            'days_since_ran', 'last_race_type', 'last_race_beaten_fav',
            'weight_pounds', 'form_type', 'adjusted_rating', 'dam_year_born',
            'sire_year_born', 'dam_sire_year_born', 'days_since_ran_type']

output = ['finish_position']

#Create the opponent attributes
op_runner = []
for op in runner:
    op_runner.append('op_'+op)

#Combine
binary_colnames = race+runner+op_runner+['outcome']

#Sample 100 competition data
sample_number = 100
sample = random.sample(list(pd.unique(data['race_id'])),sample_number)
sample_data = data[data.race_id.isin(sample)]
group = sample_data.groupby('race_id', sort=False)

#two-category transform
count = 0
binary_data = pd.DataFrame(columns = binary_colnames) 
for each_race in group:
    count += 1
    print('已完成'+str(round(count/sample_number*100,4))+'%')
    each_race = each_race[1]
    one_race_infor = each_race[race][0:1]
    one_race_runner = each_race[runner]
    one_race_outcome = each_race['finish_position']
    for runner_index in range(len(each_race)):
        other_runners = list(range(len(each_race)))
        other_runners.remove(runner_index)
        for other_runner_index in other_runners:
            outcome = 1 if one_race_outcome.values[runner_index] < one_race_outcome.values[other_runner_index] else 0
            d1 = one_race_infor.values.reshape([1,-1])
            d2 = one_race_runner.values[runner_index].reshape([1,-1])
            d3 = one_race_runner.values[other_runner_index].reshape([1,-1])
            d4 = np.array(outcome).reshape([1,-1])
            binary_data = binary_data.append(pd.DataFrame(np.hstack((d1,d2,d3,d4)),columns = binary_colnames),ignore_index=True)

#Delete race_id and runner_id
delete_var_stage2 = ['race_id','runner_id','op_runner_id']
binary_data.drop(delete_var_stage2,axis=1, inplace=True)

#Classifier the attributes
con_var = ['race_num','num_runners','num_finishers','rating','min_age',
'distance_yards','added_money','official_rating','speed_rating',
'private_handicap','winning_time_secs','standard_time_secs',
'penalty_value','prize_pos_1','prize_pos_2','prize_pos_3','prize_pos_4',
'last_winner_year','last_winner_runners','last_winner_age','last_winner_weight',
'last_winner_sp_decimal','distance_travelled','age','cloth_number',
'official_rating.1','forecast_price_decimal','starting_price_decimal',
'position_in_betting','days_since_ran','weight_pounds','adjusted_rating',
'dam_year_born','sire_year_born','dam_sire_year_born']

cate_var = ['course','race_type','going','direction','class',
'draw_advantage','handicap','all_weather','seller','claimer', 
'apprentice', 'maiden', 'amateur','country','track_type',
'advanced_going','trifecta','age_range','last_winner_bred',
'colour','gender','bred','official_rating_type','speed_rating_type',
'private_handicap_type','trainer_id','jockey_id','sire_id','dam_sire_id',
'last_race_type','last_race_beaten_fav','form_type','days_since_ran_type']

op_cate_var = []
for op in set(runner) & set(cate_var):
    op_cate_var.append('op_'+op)
all_cate_var = cate_var + op_cate_var

op_con_var = []
for op in set(runner) & set(con_var):
    op_con_var.append('op_'+op)
all_con_var = con_var + op_con_var

#Data Transforming
#[1] One-Hot
for col in [all_cate_var+['outcome']]:
    dummy = pd.get_dummies(binary_data[col],prefix = col)
    binary_data.drop(col,axis = 1,inplace = True)
    binary_data = pd.concat([binary_data, dummy], axis = 1)

#[2] MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
binary_data[all_con_var] = MMS.fit_transform(binary_data[all_con_var])

train_x = np.array(binary_data.drop(columns=['outcome_0','outcome_1']))
train_y = np.array(binary_data['outcome_1']).astype('int')
