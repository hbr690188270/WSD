import pickle 
from data_util import *
import os
import copy

with open("aux_files/tag_data.pkl",'rb') as f:
    taged_data = pickle.load(f)

if os.path.exists("aux_files/anotation.pkl"):
    with open("aux_files/anotation.pkl",'rb') as f:
        anotated_data = pickle.load(f)
else:
    anotated_data = []

if os.path.exists("scaned.pkl"):
    with open("scaned.pkl",'rb') as f:
        scaned_data = pickle.load(f)
else:
    scaned_data = []
curr_index = len(scaned_data)
# for data in all_data:
num = len(taged_data)
with open("aux_files/word_candidate.pkl",'rb') as f:
    word_candidate = pickle.load(f)
with open("aux_files/senseid.pkl",'rb') as f:
    id_dict = pickle.load(f)

pos_dic = {
    'n':'noun',
    'v':'verb', 
    'a':'adj', 
    'd':'adv', 
}

check_pos_list = ['noun', 'verb', 'adj', 'adv']

def transform_pos(pos,pos_dict):
    if pos not in pos_dict:
        return ''
    else:
        return pos_dict[pos]

for i in range(curr_index,45):
    scaned_data.append(i)
    curr_data = taged_data[i]
    sentence = [x[0] for x in curr_data]
    pos_tags = [x[1] for x in curr_data]
    print("sentence: ",''.join(sentence))
    decision = input("whether select this sentence: ")
    if decision != 'y':
        continue
    assert len(sentence) == len(pos_tags)
    for j in range(len(sentence)):
        num_taged = len(anotated_data)
        print()
        print("num_taged: ",num_taged)
        print()
        anotation_result = {}
        word_pos = pos_tags[j]
        target_word = sentence[j]
        position = j
        context = sentence.copy()
        context[position] = '<target>'
        transformed_pos = transform_pos(word_pos,pos_dic)
        word_pos_list = list(id_dict[target_word].keys())
        if transformed_pos not in check_pos_list or transformed_pos not in word_pos_list:
            continue
        anotation_result['context'] = context
        anotation_result['position'] = position
        anotation_result['target_word'] = target_word
        anotation_result['target_pos'] = transformed_pos
        context[position] = target_word
        senses = id_dict[target_word][transformed_pos]
        if len(senses) <= 1:
            continue
        str_sen = ' '.join(context)
        print("sentence: ",str_sen)
        print("word pos: ",transformed_pos)
        print("target_word: ",target_word)
        print("sense_list: ")
        for idx in range(len(senses)):
            print("\tsense %d: "%(idx), senses[idx])
        selection = input("select: ")
        try:
            sel_index = int(selection)
        except:
            print("error: input again")
            selection = input("select: ")
            sel_index = int(selection)
        if sel_index < 0:
            continue
        anotation_result['sense_index'] = sel_index
        anotated_data.append(anotation_result)
        print()
        print('-'*60)
        print()

with open("aux_files/anotation.pkl",'wb') as f:
    pickle.dump(anotated_data,f)
with open("scaned.pkl",'wb') as f:
    pickle.dump(scaned_data,f)
print(anotated_data[-1])

