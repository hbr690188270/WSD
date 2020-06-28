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
import tensorflow_hub as hub
import pandas as pd
import OpenHowNet


class Dictionary():
    def __init__(self):
        self.word_dict = {}
        # count = 0
        with open("sgns_baidubaike_bigram_char",'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                items = line.split()
                # try:
                word = ''.join(items[:-300])
                vectors = np.array([float(x) for x in items[-300:]])
                # except:
                    # print(word)
                    # print(items[:5])
                self.word_dict[word] = vectors
        keys = list(self.word_dict.keys())[:20]
        # print(self.word_dict.keys())
        for key in keys:
            print(key,self.word_dict[key][:10])

    def __call__(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return None

'''
借助词向量进行消岐
An Unsupervised Word Sense Disambiguation System for Under-Resourced Languages
'''
class dense_filter():
    def __init__(self):
        with open("vector_dict.pkl",'rb') as f:
            self.word_dict = pickle.load(f)
        # self.word_dict = Dictionary()
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

    def load_ngram_dict(self):
        with open("aux_files/ngram_dict.pkl",'rb') as f:
            ngram_dict = pickle.load(f)
        return ngram_dict

    def test_model(self):
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
            former_word = sentence[pos - 1] if pos >= 1 else None
            latter_word = sentence[pos + 1] if pos <= len(sentence) - 2 else None
            position = pos

            sentence[pos] = target_word
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
            # target_index,prob_list = self.select_sense(sentence,position,target_word,sub_dict,filter_dict,former_word,latter_word)
            target_index,prob_list = self.select_sense(sentence,target_word,sub_dict)

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

    def select_sense(self,word_sentence,orig_word,sub_dict,filter_dict = None, former_word = None,latter_word = None):
        # if filter_dict == None:
        sim_score_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            filter_list = []
            flag = 0
            pre_freq_count = 0
            aft_freq_count = 0
            if filter_dict != None:
                for subword in unique_list:
                    if former_word != None:
                        if former_word in filter_dict and subword in filter_dict[former_word]:
                            flag += 1
                            pre_freq_count += filter_dict[former_word][subword]
                    if latter_word != None:
                        if subword in filter_dict and latter_word in filter_dict[subword]:
                            flag += 1
                            aft_freq_count += filter_dict[subword][latter_word]
                    freq_sum = pre_freq_count + aft_freq_count
                    if freq_sum >= 30:
                    # if flag >= 1:
                        filter_list.append(subword)
                print("%d words are droped with ngram model"%(len(unique_list) - len(filter_list)))
                final_list = filter_list
            else:
                final_list = unique_list

            # sentence_vector = np.mean([self.word_dict[x] for x in word_sentence])
            sentence_vector = []
            for x in word_sentence:
                vec = self.word_dict(x)
                # print(vec.shape)
                if vec is not None:
                    sentence_vector.append(vec)
            sentence_vector = np.stack(sentence_vector,axis = 0).astype(np.float32)
            sentence_vector = np.mean(sentence_vector,axis = 0)
            assert sentence_vector.shape[0] == 300
            score = self.cal_synset_score(sentence_vector,final_list)
            sim_score_list.append(score)
        target_index = np.argmax(sim_score_list)
        return target_index,sim_score_list
    
    def cal_synset_score(self,sentence_vector,final_list):
        if len(final_list) == 0:
            return -1
        synset_vec = []
        for x in final_list:
            vec = self.word_dict(x)
            if vec is not None:
                synset_vec.append(vec)
        synset_vec = np.stack(synset_vec,axis = 0)
        synset_vec = np.mean(synset_vec,axis = 0)
        score = np.dot(sentence_vector,synset_vec)/np.sqrt((np.sum(np.square(sentence_vector)) * np.sum(np.square(sentence_vector))))
        print(score)
        return score


'''
Score规范化
'''
class bert_filter_2():
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

    def load_ngram_dict(self):
        with open("aux_files/ngram_dict.pkl",'rb') as f:
            ngram_dict = pickle.load(f)
        return ngram_dict

    def cal_prob(self,sentence,orig_word,sub_word,position):
        mask_char = ['[MASK]' for _ in range(len(sub_word))] 
        copy_sentence = sentence[:position] + mask_char + sentence[position + len(orig_word):]
        # print(sentence)
        # print(copy_sentence)
        # pause = input("?")
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

        char_sub = list(sub_word)
        subword_ids = self.tokenizer.convert_tokens_to_ids(char_sub)
        # print(char_sub,subword_ids)
        # for pos in range(min(len(orig_word),len(sub_word))):
        #     # print(pos + position + 1)
        #     word_id = subword_ids[pos]
        #     all_probs.append(pre_scores[pos+position + 1][word_id])

        #     subids = np.argsort(-pre_scores[pos+position+1])[:20]
        #     char_list = self.tokenizer.convert_ids_to_tokens(subids)
        #     print(char_list)
        #     print()

        for pos in range(len(sub_word)):
            # print(pos + position + 1)
            word_id = subword_ids[pos]
            all_probs.append(pre_scores[pos+position+1][word_id])
        # print(all_probs)
        # pause = input("?")
        return np.mean(all_probs)

    def predict_synset_prob(self,sentence,positions, orig_word,sub_list, drop_outlier = False):
        new_sen = sentence.copy()
        prob_list = []
        count = 0
        for idx in range(positions,positions + len(orig_word)):
            new_sen[idx] = '[MASK]'
        
        # print()
        # print("subwords: ", sub_list)
        for sub_word in sub_list:
            assert sub_word != orig_word, sub_word + " " + orig_word
            count += 1
            # print(new_sen)
            prob_list.append(self.cal_prob(new_sen,orig_word,sub_word,positions))
        prob_list = np.array(prob_list)
        if len(prob_list) >= 1:
            if drop_outlier:
                if len(prob_list) >= 20:
                    mean_value = np.mean(prob_list)
                    data = pd.DataFrame(prob_list)
                    q1 = data.quantile(q = 0.25,axis = 0)[0]
                    q2 = data.quantile(q = 0.75,axis = 0)[0]
                    # print(prob_list)
                    # print(q1,q2)
                    # pause = input("?")
                    prob_lower = mean_value
                    droped_prob_list = []
                    for prob in prob_list:
                        if prob < prob_lower:
                            continue
                        else:
                            droped_prob_list.append(prob)
                    print("%d values are droped"%(len(prob_list) - len(droped_prob_list)))
                    return droped_prob_list
                elif len(prob_list) >= 10:
                    mean_value = np.mean(prob_list)
                    data = pd.DataFrame(prob_list)
                    q1 = data.quantile(q = 0.25,axis = 0)[0]
                    q2 = data.quantile(q = 0.75,axis = 0)[0]
                    # print(prob_list)
                    # print(q1,q2)
                    # pause = input("?")
                    prob_lower = q1
                    droped_prob_list = []
                    for prob in prob_list:
                        if prob < prob_lower:
                            continue
                        else:
                            droped_prob_list.append(prob)
                    print("%d values are droped"%(len(prob_list) - len(droped_prob_list)))
                    return droped_prob_list
                else:
                    return prob_list
            else:
                print(np.mean(prob_list),prob_list,sub_list)
                return prob_list
        else:
            print(None,prob_list,sub_list)
            return prob_list

    def test_model(self,method = "norm",filter = 'ngram'):
        if filter == 'ngram':
            filter_dict = self.load_ngram_dict()
        else:
            filter_dict = None
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
            former_word = sentence[pos - 1] if pos >= 1 else None
            latter_word = sentence[pos + 1] if pos <= len(sentence) - 2 else None
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
                target_index,prob_list = self.select_sense(char_sentence,position,target_word,sub_dict,filter_dict,former_word,latter_word)
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

    def select_sense(self,sentence, positions, orig_word,sub_dict,filter_dict = None,former_word = None,latter_word = None):
        avg_prob_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            filter_list = []
            flag = 0
            pre_freq_count = 0
            aft_freq_count = 0
            if filter_dict != None:
                for subword in unique_list:
                    if former_word != None:
                        if former_word in filter_dict and subword in filter_dict[former_word]:
                            flag += 1
                            pre_freq_count += filter_dict[former_word][subword]
                    if latter_word != None:
                        if subword in filter_dict and latter_word in filter_dict[subword]:
                            flag += 1
                            aft_freq_count += filter_dict[subword][latter_word]
                    freq_sum = pre_freq_count + aft_freq_count
                    if freq_sum >= 30:
                    # if flag >= 1:
                        filter_list.append(subword)
                print("%d words are droped with ngram model"%(len(unique_list) - len(filter_list)))
                final_list = filter_list
            else:
                final_list = unique_list

            prob_list = self.predict_synset_prob(sentence,positions,orig_word,final_list)
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


'''
计算LM loss的版本
'''
class bert_filter_new():
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

    def load_ngram_dict(self):
        with open("aux_files/ngram_dict.pkl",'rb') as f:
            ngram_dict = pickle.load(f)
        return ngram_dict

    def cal_prob(self,sentence,orig_word,sub_word,position,only_target = True):
        char_sub = list(sub_word)
        copy_sentence = sentence[:position] + char_sub + sentence[position + len(orig_word):]
        copy_sentence = ['[CLS]'] + copy_sentence
        copy_sentence += ['[SEP]']
        # pos_with_spe = position + 1
        text = ' '.join(copy_sentence) 
        bert_tokens = self.tokenizer.tokenize(text)
        # print(sentence)
        # print(bert_tokens)
        id_list = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        if only_target:
            input_ids = torch.tensor([id_list]).to("cuda")
            lm_list = np.array(id_list.copy())
            lm_list[1:position + 1] = -100
            lm_list[position + len(orig_word) + 1:-1] = -100
            lm_labels = torch.tensor([lm_list]).to('cuda')
        else:
            input_ids = torch.tensor([id_list]).to('cuda')
            lm_labels = input_ids
        
        # print(copy_sentence)
        # print(lm_list)
        # print(input_ids)
        # pause = input("?")

        outputs = self.bert_model(input_ids,masked_lm_labels = lm_labels)
        loss, prediction_scores = outputs[:2]
        lm_loss = loss.item()
        return lm_loss

    def predict_synset_prob(self,sentence,position, orig_word,sub_list, drop_outlier = False):
        new_sen = sentence.copy()
        prob_list = []
        count = 0
        # for idx in range(position,position + len(orig_word)):
        #     new_sen[idx] = '[MASK]'
        
        # print()
        # print("subwords: ", sub_list)
        for sub_word in sub_list:
            assert sub_word != orig_word, sub_word + " " + orig_word
            count += 1
            # print(new_sen)
            prob_list.append(self.cal_prob(new_sen,orig_word,sub_word,position))
        prob_list = np.array(prob_list)
        if len(prob_list) >= 1:
            if drop_outlier:
                if len(prob_list) >= 20:
                    mean_value = np.mean(prob_list)
                    data = pd.DataFrame(prob_list)
                    q1 = data.quantile(q = 0.25,axis = 0)[0]
                    q2 = data.quantile(q = 0.75,axis = 0)[0]
                    # print(prob_list)
                    # print(q1,q2)
                    # pause = input("?")
                    loss_higher = q2
                    droped_prob_list = []
                    for prob in prob_list:
                        if prob > loss_higher:
                            continue
                        else:
                            droped_prob_list.append(prob)
                    print("%d values are droped"%(len(prob_list) - len(droped_prob_list)))
                    return droped_prob_list
                elif len(prob_list) >= 10:
                    mean_value = np.mean(prob_list)
                    data = pd.DataFrame(prob_list)
                    q1 = data.quantile(q = 0.25,axis = 0)[0]
                    q2 = data.quantile(q = 0.75,axis = 0)[0]
                    # print(prob_list)
                    # print(q1,q2)
                    # pause = input("?")
                    prob_lower = q1
                    droped_prob_list = []
                    for prob in prob_list:
                        if prob < prob_lower:
                            continue
                        else:
                            droped_prob_list.append(prob)
                    print("%d values are droped"%(len(prob_list) - len(droped_prob_list)))
                    return droped_prob_list
                else:
                    return prob_list
            else:
                print(np.mean(prob_list),prob_list,sub_list)
                return prob_list
        else:
            print(None,prob_list,sub_list)
            return prob_list

    def test_model(self,method = "norm",filter = 'ngram'):
        data_list = self.load_all_words_data()
        if filter == 'ngram':
            filter_dict = self.load_ngram_dict()
        else:
            filter_dict = None

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
            former_word = sentence[pos - 1] if pos >= 1 else None
            latter_word = sentence[pos + 1] if pos <= len(sentence) - 2 else None
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
                # target_index,prob_list = self.select_sense(char_sentence,position,target_word,sub_dict,filter_dict,former_word,latter_word)
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

    def select_sense(self,sentence, positions, orig_word,sub_dict,filter_dict = None,former_word = None,latter_word = None):
        avg_prob_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            filter_list = []
            flag = 0
            pre_freq_count = 0
            aft_freq_count = 0
            if filter_dict != None:
                for subword in unique_list:
                    if former_word != None:
                        if former_word in filter_dict and subword in filter_dict[former_word]:
                            flag += 1
                            pre_freq_count += filter_dict[former_word][subword]
                    if latter_word != None:
                        if subword in filter_dict and latter_word in filter_dict[subword]:
                            flag += 1
                            aft_freq_count += filter_dict[subword][latter_word]
                    freq_sum = pre_freq_count + aft_freq_count
                    if freq_sum >= 30:
                    # if flag >= 1:
                        filter_list.append(subword)
                print("%d words are droped with ngram model"%(len(unique_list) - len(filter_list)))
                final_list = filter_list
            else:
                final_list = unique_list

            prob_list = self.predict_synset_prob(sentence,positions,orig_word,final_list)
            if len(prob_list) >= 1:
                avg_prob_list.append(np.mean(prob_list))
            else:
                avg_prob_list.append(10000)
        target_index = np.argmin(avg_prob_list)
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
  

'''
ACL2017
'''
class SAT_filter():
    def __init__(self,window_size = 8, model_path = 'acl17_files/'):
        self.modle_path = model_path
        self.window_size = window_size
        self.word_dict = self.read_vecfiles("word")
        self.sememe_dict = self.read_vecfiles("sememe")
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.load_dict()

    def read_vecfiles(self,file):
        vec_dict = {}
        with open(self.modle_path + file + '-vec.txt','r',encoding = 'utf-8') as f:
            for line in f:
                items = line.split()
                word = items[0]
                vec = np.array([float(x) for x in items[1:]])
                if len(vec) != 300:
                    continue
                vec_dict[word] = vec
        return vec_dict
   
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


    def test_model(self):
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
            position = curr_data['position']
            word_pos = curr_data['target_pos']
            sentence[position] = '<target>'
            print(curr_data['position'],target_word)
            context_words = []
            for idx in range(max([0,position - self.window_size]), min([len(sentence),position + self.window_size + 1])):
                if idx == position:
                    continue
                context_words.append(sentence[idx])

            if sense_key == -1:
                continue
            if word_pos not in check_pos_list:
                print("pos not in valid pos list: ", word_pos)
                continue

            sub_dict = self.word_candidate[target_word][word_pos]
            if len(sub_dict) <= 1:
                continue
            # print(sentence)
            print(target_word,word_pos)
            print(position)
            for idx, subwords in sub_dict.items():
                print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            sense_dict = self.word_sense_id_sem[target_word][word_pos]
            target_index,prob_list = self.select_sense(context_words,target_word,sense_dict)
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
    
    def select_sense(self,context_words,target_word,sense_dict):
        context_embedding = []
        for word in context_words:
            if word in self.word_dict:
                context_embedding.append(self.word_dict[word])
            else:
                print("word ",word,"not in the dict!")
        if len(context_embedding) == 0:
            return -1
        context_embedding = np.mean(np.stack(context_embedding,axis = 0),axis = 0)
        sim_score_list = []
        for i in range(len(sense_dict)):
            sense_set = sense_dict[i]
            sense_embeddings = []
            for sem in sense_set:
                if sem not in self.sememe_dict:
                    continue
                sense_embeddings.append(self.sememe_dict[sem])
            if len(sense_embeddings) == 0:
                sim_score_list.append(-1)
            else:
                sense_embeddings = np.mean(np.stack(sense_embeddings,axis = 0),axis = 0)
                sim_score = self.cos_sim(context_embedding,sense_embeddings)
                sim_score_list.append(sim_score)
        target_index = np.argmax(sim_score_list)
        return target_index, sim_score_list

    def cos_sim(self,vec1,vec2):
        return np.dot(vec1,vec2)/np.sqrt(np.sum(np.square(vec1)) * np.sum(np.square(vec2)))




'''
中文信息学报2015
'''
class cip15_filter():
    def __init__(self,model_path = 'cip15_model/',vec_dim = 300,window_size = 8):
        self.vec_dim = vec_dim
        self.model_path = model_path
        self.window_size = window_size
        self.word_vec_dict, self.sem_vec_dict = self.load_vector()
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.hownet_dict = OpenHowNet.HowNetDict()
        self.load_dict()
    
    def load_vector(self):
        with open(self.model_path + "word_vec.pkl",'rb') as f:
            word_vec = pickle.load(f)
        with open(self.model_path + "sem_vec.pkl",'rb') as f:
            sem_vec = pickle.load(f)
        return word_vec,sem_vec

    def get_annotation(self,word):
        try:
            sememes = self.hownet_dict.get_sememes_by_word(word,structured = False,lang = 'zh',merge = True)
        except:
            return None
        if len(sememes) == 1:
            sememe_list = []
            for sem in sememes:
                sememe_list.append(sem)
            return sememe_list[0]
        else:
            return None


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

    def test_model(self):
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
            position = curr_data['position']
            word_pos = curr_data['target_pos']
            sentence[position] = '<target>'
            print(curr_data['position'],target_word)
            context_words = []
            for idx in range(max([0,position - self.window_size]), min([len(sentence),position + self.window_size + 1])):
                if idx == position:
                    continue
                context_words.append(sentence[idx])

            if sense_key == -1:
                continue
            if word_pos not in check_pos_list:
                print("pos not in valid pos list: ", word_pos)
                continue

            sub_dict = self.word_candidate[target_word][word_pos]
            if len(sub_dict) <= 1:
                continue
            # print(sentence)
            print(target_word,word_pos)
            print(position)
            for idx, subwords in sub_dict.items():
                print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            sense_dict = self.word_sense_id_sem[target_word][word_pos]
            target_index,prob_list = self.select_sense(context_words,target_word,sense_dict)
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

    def select_sense(self,context_words,target_word,sense_dict):
        context_embedding = []
        for word in context_words:
            if word in self.word_vec_dict:
                context_embedding.append(self.word_vec_dict[word])
            else:
                annotation = self.get_annotation(word)
                if annotation == None:
                    print("word ",word,"not in the dict!")
                elif annotation in self.sem_vec_dict:
                    print(word,"with only one sememe: ",annotation)
                    context_embedding.append(self.sem_vec_dict[annotation])
        if len(context_embedding) == 0:
            return -1
        context_embedding = np.mean(np.stack(context_embedding,axis = 0),axis = 0)
        sim_score_list = []
        for i in range(len(sense_dict)):
            sense_set = sense_dict[i]
            sense_embeddings = []
            for sem in sense_set:
                if sem not in self.sem_vec_dict:
                    continue
                sense_embeddings.append(self.sem_vec_dict[sem])
            if len(sense_embeddings) == 0:
                sim_score_list.append(-1)
            else:
                sense_embeddings = np.mean(np.stack(sense_embeddings,axis = 0),axis = 0)
                sim_score = self.cos_sim(context_embedding,sense_embeddings)
                sim_score_list.append(sim_score)
        target_index = np.argmax(sim_score_list)
        return target_index, sim_score_list

    def cos_sim(self,vec1,vec2):
        return np.dot(vec1,vec2)/np.sqrt(np.sum(np.square(vec1)) * np.sum(np.square(vec2)))
