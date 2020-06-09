import tensorflow as tf
import os
import numpy as np
import copy
import pickle
import math
import torch
from tokenization import tokenization
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from train.modeling import get_lm_loss,GroverConfig,GroverModel
from keras.backend.tensorflow_backend import set_session
from sklearn.cluster import MeanShift, estimate_bandwidth 

class gpt2_model():
    def __init__(self):
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.build_model()

    def build_model(self):
        self.input_ids = tf.placeholder(shape = [1,None],dtype = tf.int32)
        news_config = GroverConfig.from_json_file('configs/mega.json')
        self.model = GroverModel(config = news_config,is_training = False, input_ids = self.input_ids)
        self.loss = self.model.lm_loss()
        self.load_dict()
        vocab_file_path = 'gpt2/vocab.txt'
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)

        
    def load_all_words_data(self):
        with open("aux_files/anotation.pkl",'rb') as f:
            all_data = pickle.load(f)
        return all_data
    
    def load_dict(self):
        with open("aux_files/senseid.pkl",'rb') as f:
            self.word_sense_id_sem = pickle.load(f)
        with open("aux_files/word_candidate.pkl",'rb') as f:
            self.word_candidate = pickle.load(f)
        # with open("word_sem")

    def select_sense(self,sentence,position,synsets,sess):
        avg_ppl_list = []
        for i in range(len(synsets)):
            subwords = synsets[i]
            ppl_list = self.predict_synset_ppl(sentence,position,subwords,sess)
            if len(ppl_list) >= 1:
                avg_ppl_list.append(np.mean(ppl_list))
            else:
                avg_ppl_list.append(100000)
        target_index = np.argmin(avg_ppl_list)
        print("selection: %d"%(target_index))
        print()
        return target_index

    def select_sense_mini(self,sentence, position, synsets, sess):
        min_ppl_list = []
        for i in range(len(synsets)):
            subwords = synsets[i]
            ppl_list = self.predict_synset_ppl(sentence,position,subwords,sess)
            if len(ppl_list) >= 1:
                min_ppl_list.append(np.min(ppl_list))
            else:
                min_ppl_list.append(100000)
        target_index = np.argmin(min_ppl_list)
        print("selection: %d"%(target_index))
        print()
        return target_index

    def select_sense_cluster(self,sentence, position,synsets,sess):
        cluster_ppl_list = []
        for i in range(len(synsets)):
            subwords = synsets[i]
            ppl_list = self.predict_synset_ppl(sentence,position,subwords,sess)
            cluster_ppl_list.append(self.cluster_list(ppl_list,bandwidth = 1))
        target_index = np.argmin(cluster_ppl_list)
        print("selection: %d"%(target_index))
        print()
        return target_index

    def cluster_list(self,ppl_list,bandwidth = -1):
        if bandwidth <= 0:
            cluster_model = MeanShift()
        else:
            cluster_model = MeanShift(bandwidth = bandwidth)
        label_list = cluster_model.fit_predict(np.array(ppl_list).reshape(-1,1))
        group_num = np.max(label_list) + 1
        if group_num == 1:
            return np.mean(ppl_list)
        else:
            ppl_dict = {}
            for i in range(len(label_list)):
                label = label_list[i]
                if label not in ppl_dict:
                    ppl_dict[label] = []
                ppl_dict[label].append(ppl_list[i])
            min_index = -1
            min_ppl = 1000000
            for i in range(len(ppl_dict)):
                avg_ppl = np.mean(ppl_dict[i])
                if avg_ppl < min_ppl:
                    min_ppl = avg_ppl
                    min_index = i
            return np.mean(ppl_dict[min_index])
        # return label_list

    def cal_ppl(self,sentence,sess):
        text = ' '.join(sentence) 
        line = tokenization.convert_to_unicode(text)
        bert_tokens = self.tokenizer.tokenize(line)
        encoded = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        # tokenize_input = sentence
        
        lm_loss = sess.run([self.loss],
                                 feed_dict = {self.input_ids:np.array([encoded])} 
                                        )
        # loss, logits = outputs[:2]
        sentence_prob = lm_loss[0]
        ppl = sentence_prob
        return ppl

    def predict_synset_ppl(self,sentence,position, sub_list,sess):
        new_sen = sentence.copy()
        ppl_list = []
        count = 0
        sentence_set = []
        # length = 0
        for word in sub_list:
            count += 1
            new_sen[position] = word
            # print(new_sen)
            ppl_list.append(math.exp(self.cal_ppl(new_sen,sess)))
        # ppl /= count
        # print(ppl,sub_list)
        return ppl_list

    def test_model(self,sess):
        data_list = self.load_all_words_data()
        pos_dict = {
                'n':'noun',
                'v':'verb', 
                'a':'adj', 
                'd':'adv', 
            }
        check_pos_list = ['noun', 'verb', 'adj', 'adv']
        for i in range(len(data_list)):
            curr_data = data_list[i]
            # print(curr_data)
            if len(curr_data) == 0:
                continue
            sentence = curr_data['context']
            word = curr_data['target_word']
            sense_key = curr_data['sense_index']
            position = curr_data['position']
            word_pos = curr_data['target_pos']
            pos = word_pos
            if sense_key == -1:
                continue
            if pos not in check_pos_list:
                print("pos not in valid pos list: ", pos)
                continue

            # subwords,synset_list = self.get_subwords(word,pos)
            sub_dict = self.word_candidate[word][pos]
            # if len(sub_dict) <= 1:
            #     continue
            print(sentence)
            print(word,pos)
            print(position)
            for idx, subwords in sub_dict.items():
                print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            target_index = self.select_sense_cluster(sentence,position,sub_dict,sess)
            print(target_index, sense_key)
            print()
            print("select sense: ", self.word_sense_id_sem[word][pos][target_index])
            print("real sense:", self.word_sense_id_sem[word][pos][sense_key])

            if target_index == sense_key:
                print("!")
                self.correct += 1
            self.word_count += 1
            print()
            print('-'*60)
            print()
        print(self.correct,self.word_count)
        print(self.word_count,self.sense_num)
        # print(self.rand_count,self.word_count)








