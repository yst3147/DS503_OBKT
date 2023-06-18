from tqdm import tqdm
import pandas as pd
from collections import Counter 


q_t = pd.read_pickle("./data/ednet/preprocessed_total_data.pkl")
q_t['ind'] = q_t.index

def list_to_string(lst):
    return ', '.join(map(str, lst))

def frequency_ratio(lst,q_c_a):

    freq_dict = Counter(lst)

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
filtered_q_t = q_t.groupby('question_id').filter(lambda x: x[(x.corretness == 1)]['correct_answer'].nunique() <= 1 and x[(x.corretness == 0)]['correct_answer'].nunique() <= 1)
q_t = filtered_q_t

for q in tqdm(q_t.question_id.unique()):
        
    q_df = q_t[q_t.question_id == q].dropna()
    q_u_a = q_df.user_answer.tolist()
    q_c_a = q_df.correct_answer.tolist()
    if len(q_u_a):
        q_df['OptionWeight'] = frequency_ratio(q_u_a, q_c_a)
        result_list.append(q_df)

result_df = pd.concat(result_list).sort_values(by=['ind'], axis=0)

result_df = result_df[['user_id', 'question_id', 'user_answer', 'correct_answer', 'tags', 'corretness', 'option_score', 'OptionWeight']]
result_df.to_pickle("../data/ednet/method_1/ednet.pkl")