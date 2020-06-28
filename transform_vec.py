import numpy as np 
import OpenHowNet
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pickle

# Word2Vec()

wv_model = Word2Vec.load("cip15_model/word2vec_model_2")

word_vec_dict = {}
sem_vec_dict = {}

hownet_dict = OpenHowNet.HowNetDict()
def get_annotation(word):
    sememes = hownet_dict.get_sememes_by_word(word,structured = False,lang = 'zh',merge = True)
    if len(sememes) == 1:
        sememe_list = []
        for sem in sememes:
            sememe_list.append(sem)
        return sememe_list[0]
    else:
        return None

sem_list = []
for word,Vocab in wv_model.wv.vocab.items():
    vec = wv_model.wv.vectors[Vocab.index]
    if len(word) > 2 and word[:2] == 's_':
        sem = word[2:]
        print(sem)
        sem_list.append(sem)
        sem_vec_dict[sem] = vec
        continue
    word_vec_dict[word] = vec
    # sem = get_annotation(word)
    # if sem == None:
    #     continue
    # else:
    #     word_vec_dict[sem] = vec

print(len(sem_list))
with open("cip15_model/word_vec.pkl",'wb') as f:
    pickle.dump(word_vec_dict,f)

with open("cip15_model/sem_vec.pkl",'wb') as f:
    pickle.dump(sem_vec_dict,f)















