import tensorflow as tf
import os
import numpy as np
import copy
import pickle
import math
import torch
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import BertForMaskedLM, BertTokenizer
from sklearn.cluster import MeanShift, estimate_bandwidth 


class bert_filter():
    def __init__(self,model_type = 'bert-base-chinese'):
        self.bert_model = BertForMaskedLM.from_pretrained(model_type,
                                                            cache_dir = '/data2/private/houbairu/model_cache/bert-chinese/').to("cuda")
        self.bert_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_type,cache_dir = '/data2/private/houbairu/model_cache/bert-chinese')
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.load_dict()
        
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


    def cal_prob(self,sentence,orig_word,sub_word,position):
        copy_sentence = sentence[:]
        copy_sentence = ['[CLS]'] + copy_sentence
        copy_sentence += ['[SEP]']
        # pos_with_spe = position + 1
        text = ' '.join(copy_sentence) 
        bert_tokens = self.tokenizer.tokenize(text)
        # print(sentence)
        # print(bert_tokens)
        id_list = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        input_ids = torch.tensor([id_list]).to("cuda")
        outputs = self.bert_model(input_ids,masked_lm_labels = input_ids)
        pre_scores = outputs[1].detach().cpu().numpy()[0]
        all_probs = []

        char_sub = []
        for ch in sub_word:
            char_sub.append(ch)
        subword_ids = self.tokenizer.convert_tokens_to_ids(char_sub)
        # print(char_sub,subword_ids)
        if len(orig_word) == len(sub_word):
            for pos in range(len(sub_word)):
                # print(pos + position + 1)
                word_id = subword_ids[pos]
                all_probs.append(pre_scores[pos+position+1][word_id])
        elif len(orig_word) > len(sub_word):
            for pos in range(len(sub_word)):
                # print(pos + position + 1)
                word_id = subword_ids[pos]
                all_probs.append(pre_scores[pos+position+1][word_id])
        else:
            for pos in range(len(orig_word)):
                # print(pos + position + 1)
                word_id = subword_ids[pos]
                all_probs.append(pre_scores[pos+position + 1][word_id])
            for pos in range(len(orig_word),len(sub_word)):
                word_id = subword_ids[pos]
                all_probs.append(pre_scores[position + len(orig_word)][word_id])
        # print(all_probs)
        # pause = input("?")
        return np.mean(all_probs)

    def predict_synset_prob(self,sentence,positions, orig_word,sub_list):
        new_sen = sentence.copy()
        prob_list = []
        count = 0
        for idx in range(positions,positions + len(orig_word)):
            new_sen[idx] = '[MASK]'
        for sub_word in sub_list:
            assert sub_word != orig_word, sub_word + " " + orig_word
            count += 1
            # print(new_sen)
            prob_list.append(self.cal_prob(new_sen,orig_word,sub_word,positions))
        # if count == 0:
        #     return 0
        # else:
        #     return np.mean(prob_list)
        # ppl /= count
        if len(prob_list) >= 1:
            print(np.mean(prob_list),prob_list,sub_list)
        else:
            print(None,prob_list,sub_list)
        return prob_list

    def test_model(self,method = "norm"):
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
            target_word = curr_data['target_word']
            sense_key = curr_data['sense_index']
            pos = curr_data['position']
            word_pos = curr_data['target_pos']
            sentence[pos] = '<target>'
            print(curr_data['position'],target_word)
            position = 0
            for word in sentence:
                if word != '<target>':
                    position += len(word)
                else:
                    break
            # positions = [x for x in range(position,position+len(target_word))]
            # positions = position
            # pos = word_pos
            # print(sentence)
            # print(position)
            sentence[pos] = target_word
            char_sentence = []
            for ch in ''.join(sentence):
                char_sentence.append(ch)
            print(char_sentence)

            # print(positions)
            # pause = input("?")
            if sense_key == -1:
                continue
            if word_pos not in check_pos_list:
                print("pos not in valid pos list: ", word_pos)
                continue

            sub_dict = self.word_candidate[target_word][word_pos]
            # if len(sub_dict) <= 1:
            #     continue
            # print(sentence)
            print(target_word,word_pos)
            print(position)
            for idx, subwords in sub_dict.items():
                print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            if method == 'norm':
                target_index,prob_list = self.select_sense(char_sentence,position,target_word,sub_dict)
            elif method == "max":
                target_index,prob_list = self.select_sense_max(char_sentence,position,target_word,sub_dict)
            else:
                target_index,prob_list = self.select_sense_cluster(char_sentence,position,target_word,sub_dict)


            print(target_index, sense_key)
            print()
            print("select sense: ", self.word_sense_id_sem[target_word][word_pos][target_index]," prob: ",prob_list[target_index])
            print("real sense:", self.word_sense_id_sem[target_word][word_pos][sense_key]," prob: ",prob_list[sense_key])

            if target_index == sense_key:
                print("!")
                self.correct += 1
            self.word_count += 1
            print()
            print('-'*60)
            print()
            # pause = input("continue? ")
        print(self.correct,self.word_count)
        print(self.word_count,self.sense_num)

    def select_sense(self,sentence, positions, orig_word,sub_dict):
        avg_prob_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            prob_list = self.predict_synset_prob(sentence,positions,orig_word,unique_list)
            if len(prob_list) >= 1:
                avg_prob_list.append(np.mean(prob_list))
            else:
                avg_prob_list.append(-10)
        target_index = np.argmax(avg_prob_list)
        print("selection: %d"%(target_index))
        print()
        return target_index,avg_prob_list

    def select_sense_max(self,sentence, position, orig_word,sub_dict):
        max_prob_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            prob_list = self.predict_synset_prob(sentence,position,orig_word,unique_list)

            if len(prob_list) >= 1:
                max_prob_list.append(np.max(prob_list))
            else:
                max_prob_list.append(-100)
        target_index = np.argmax(max_prob_list)
        print("selection: %d"%(target_index))
        print()
        return target_index,max_prob_list

    def select_sense_cluster(self,sentence, position,orig_word,sub_dict):
        cluster_prob_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            prob_list = self.predict_synset_prob(sentence,position,orig_word,unique_list)
            if len(prob_list) == 0:
                cluster_prob_list.append(-100)
            else:
                cluster_prob_list.append(self.cluster_list(prob_list,bandwidth = 0.5))
        target_index = np.argmax(cluster_prob_list)
        print("selection: %d"%(target_index))
        print()
        return target_index,cluster_prob_list

    def cluster_list(self,prob_list,bandwidth = -1):
        if bandwidth <= 0:
            cluster_model = MeanShift()
        else:
            cluster_model = MeanShift(bandwidth = bandwidth)
        label_list = cluster_model.fit_predict(np.array(prob_list).reshape(-1,1))
        group_num = np.max(label_list) + 1
        if group_num == 1:
            return np.mean(prob_list)
        else:
            prob_dict = {}
            for i in range(len(label_list)):
                label = label_list[i]
                if label not in prob_dict:
                    prob_dict[label] = []
                prob_dict[label].append(prob_list[i])
            max_index = -1
            max_prob = -100
            for i in range(len(prob_dict)):
                avg_prob = np.mean(prob_dict[i])
                if avg_prob > max_prob:
                    max_prob = avg_prob
                    max_index = i
            return np.mean(prob_dict[max_index])
        # return label_list




