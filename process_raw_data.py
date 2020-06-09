import numpy as np 
import pickle

line_to_read = 10000
raw_data = []
count = 0
with open("zhihu_data/raw_data.txt",'r',encoding = 'utf-8') as f:
    for line in f:
        count += 1
        raw_data.append(line.strip())
        if count >= line_to_read:
            break

def process_sentence(sentence,max_len):
    selected = []
    data = sentence.split()
    len_sum = 0
    for piece in data:
        len_sum += len(piece)
        if len_sum <= max_len:
            selected.append(piece)
        else:
            break
    if len(selected) >= 1:
        return '，'.join(selected)
    else:
        return None

processed_data = []
for sentence in raw_data:
    result = process_sentence(sentence,max_len = 100)
    if result == None:
        continue
    else:
        processed_data.append(result)

with open("zhihu_data/processed_data.pkl",'wb') as f:
    pickle.dump(processed_data,f)




