from tqdm import tqdm
import pandas as pd
from collections import Counter 
import numpy as np
import datetime
import time

q_t = pd.read_csv("./data/eedi_a/train_task_1_2.csv")
q_t['ind'] = q_t.index
q_t.AnswerValue  = q_t.AnswerValue.replace([1,2,3,4],['a','b','c','d'])
q_t.CorrectAnswer  = q_t.CorrectAnswer.replace([1,2,3,4],['a','b','c','d'])

def list_to_string(lst):
    return ', '.join(map(str, lst))

def id_dic_construction(x):
    corresponding_dic = {}
    for dic_index in range(len(x)):
        corresponding_dic[x[dic_index]] = dic_index
    return corresponding_dic

def frequency_ratio(lst,q_c_a):

    freq_dict = {}
    for val in lst:
        freq_dict[val] = freq_dict.get(val, 0) + 1

    total_count = len(lst)
    ratio_dict = {}
    for val in set(['a', 'b', 'c', 'd']):
        ratio_dict[val] = freq_dict.get(val, 0) / total_count

    ratio_dict[q_c_a[0]] = 1
    sorted_dict = dict(sorted(ratio_dict.items(), key=lambda item: item[1], reverse=True))

    zero_value =[key for key, value in sorted_dict.items() if value == 0]

    max_value = max(sorted_dict.values())
    min_value = min(sorted_dict.values())
    diff = max_value - min_value

    normalized_dict = {}
    for i, (key, value) in enumerate(sorted_dict.items()):
        if i == 0:
            normalized_dict[key] = 4
        elif i == 1:
            normalized_dict[key] = 3
        elif i == 2:
            normalized_dict[key] = 2
        else:
            normalized_dict[key] = 1
    for zero in zero_value:
        normalized_dict[zero] = 1 

    normalized_list = [normalized_dict.get(val, 0) for val in lst]
    return normalized_list

result_list = []
null_list = []
filtered_q_t = q_t.groupby('QuestionId').filter(lambda x: x[(x.IsCorrect == 1)]['CorrectAnswer'].nunique() <= 1 and x[(x.IsCorrect == 0)]['CorrectAnswer'].nunique() <= 1)
null_list = q_t[~q_t.index.isin(filtered_q_t.index)]['QuestionId'].unique().tolist()
q_t = filtered_q_t

for q in tqdm(q_t.QuestionId.unique()):
    q_df = q_t[q_t.QuestionId == q].dropna()
    q_u_a = q_df.AnswerValue.tolist()
    q_c_a = q_df.CorrectAnswer.tolist()
    if len(q_u_a):
        q_df['OptionWeight'] = frequency_ratio(q_u_a, q_c_a)
        result_list.append(q_df)
result_df = pd.concat(result_list).sort_values(by=['ind'], axis=0)

# Make question_info 
question_info = pd.read_csv('./data/eedi_a/question_metadata_task_1_2.csv')
print(question_info.columns)
question_info = question_info.drop_duplicates()
for q in tqdm(null_list):
    question_info = question_info.drop(question_info[question_info['QuestionId'] == q].index)
question_info = question_info.values

# Make question_id
question_id = np.unique(question_info[:, 0])
question_dic = id_dic_construction(question_id)

# Make skill_id 
for i, kc in enumerate(tqdm(question_info)):
    question_info[i, 1] = list(map(int, kc[1][1:-1].split(',')))
skill = []
for i in tqdm(range(len(question_info))):
    skill += question_info[i, 1]
skill = np.unique(np.array(skill).astype('int64'))
skill_dic = id_dic_construction(skill)

# question, skill renaming with dictionary
for i in tqdm(range(len(question_info))):
    question_info[i, 0] = question_dic[question_info[i, 0]]
    for n in range(len(question_info[i, 1])):
        question_info[i, 1][n] = skill_dic[question_info[i, 1][n]]
question_info

# result dataframe question renaming
result_df.iloc[:, 0] = result_df.iloc[:, 0].map(question_dic)

question_df = pd.DataFrame([(question[0], question[1]) for question in question_info], columns=["QuestionId", "SubjectId"])



result_df = pd.merge(result_df, question_df, how='inner', on='QuestionId')
result_df = result_df[['UserId', 'QuestionId', 'AnswerValue', 'AnswerId', 'CorrectAnswer', 'SubjectId', 'IsCorrect', 'OptionWeight']]
result_df.columns = ['UserId', 'QuestionId', 'AnswerValue', 'AnswerId', 'CorrectAnswer', 'tag', 'IsCorrect', 'OptionWeight']

# time merge
answer_info = pd.read_csv('./data/eedi_a/answer_metadata_task_1_2.csv')
tmp_answer_info = answer_info[['AnswerId', 'DateAnswered']]
result_df = pd.merge(result_df, tmp_answer_info, how='inner', on='AnswerId')

eedi_time = list(result_df['DateAnswered'])
for i, timestr in enumerate(tqdm(eedi_time)):    
    eedi_time[i] = time.mktime(datetime.datetime.strptime(str(timestr[:-4]), "%Y-%m-%d %H:%M:%S").timetuple())
result_df['DateAnswered'] = eedi_time
result_df = result_df.sort_values(by=['DateAnswered'])
result_df.reset_index(drop = True, inplace = True)

result_df.to_pickle("../data/eedi_a/method_1/eedi_a.pkl")