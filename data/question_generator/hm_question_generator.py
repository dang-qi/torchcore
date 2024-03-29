import numpy as np
import random
import pandas as pd
import os
from .question_generator import QuestionGenerator

class HMQuestionGenerator(QuestionGenerator):
    def __init__(self, data_path, root_path, templates, max_question_number_per_item, yes_question_ratio=0.5) -> None:
        self.root_path = root_path
        self.data_path = os.path.expanduser(data_path)
        self.load_dataset()
        self.question_keywords_pool = self.generate_question_keywords_pool()
        self.templates = templates
        self.template_num = len(templates)
        self.current_question_id = 1
        assert self.template_num >= max_question_number_per_item

        self.max_question_number_per_item = max_question_number_per_item
        self.yes_question_ratio = yes_question_ratio

    def load_dataset(self):
        '''convert data as a list of dict, each dict is one item'''
        self.data = pd.read_csv(self.data_path, dtype={'article_id':str})
        self.add_image_path()

        # remove the ones that doesn't has image
        self.remove_empty_path()

        # load data from csv
        data_out = []
        for i in range(len(self.data)):
            item = {}
            for k in self.data.keys():
                item[k] = self.data[k].iat[i]
            data_out.append(item)
        self.converted_data = data_out

    def sample_templates(self):
        idxs = np.random.choice(self.template_num, self.max_question_number_per_item,replace=False)
        out = [self.templates[i] for i in idxs]
        return out

    def generate(self):
        train,val,test = self.split_dataset(self.converted_data)
        train_set = self.generate_part(train)
        val_set = self.generate_part(val)
        test_set = self.generate_part(test)
        return train_set,val_set,test_set

    def generate_part(self, data_part, reset_question_id=False):
        out = []
        if reset_question_id:
            question_id = 1
        else:
            question_id = self.current_question_id
        #test_time = 3
        i = 0
        for item in data_part:
            templates = self.sample_templates()
            for t in templates:
                one_out = {}
                one_out['image_path'] = item['path']
                one_out['article_id'] = item['article_id']
                if t['question_type'] != 'binary':
                    key_dict = {k:item[k] for k in t['keywords']}
                    #one_out['question'] = t['question'].format({k:item[k] for k in t['keywords']})
                    one_out['question'] = t['question'].format(**key_dict)
                    #one_out['concept'] = t['answer'].format(**{k:item[k] for k in t['answer_keywords']})
                    one_out['concept'] = [w.replace('_', ' ') for w in t['answer_keywords']]
                    one_out['answer'] = t['answer'].format(**{k:item[k] for k in t['answer_keywords']})
                else:
                    #yes question
                    if(np.random.random()<self.yes_question_ratio):
                        one_out['question'] = t['question'].format(**{k:item[k] for k in t['keywords']})
                        one_out['answer'] = 'yes'
                    else:
                        #one_out['question'] = t['question'].format({k:item[k] for k in t['keywords']})
                        # we need a pool for no question for each keyword and random pick one wrong answer
                        answers = [item[k] for k in t['keywords']]
                        wrong_answers = self.sample_wrong_answer_from_pool(t['keywords'], answers)
                        one_out['question'] = t['question'].format(**{k:item[k] for k in wrong_answers})
                        one_out['answer'] = 'no'

                    #one_out['concept'] = t['keywords']
                    one_out['concept'] = [w.replace('_', ' ') for w in t['keywords']]
                one_out['question_id'] = question_id
                question_id += 1
                out.append(one_out)
            i+=1
            #if i >=test_time:
            #    break
        self.current_question_id = question_id
        return out

    def split_dataset(self, all_data,ratio=(0.8,0.1,0.1),seed=0):
        assert sum(ratio)==1
        np.random.seed(seed)
        data_num = len(all_data)
        inds = np.arange(len(all_data))
        np.random.shuffle(inds)
        ind_split_train = int(np.ceil(data_num*ratio[0]))
        ind_split_val = int(np.ceil(data_num*(ratio[0]+ratio[1])))
        train_inds = inds[0:ind_split_train]
        val_inds = inds[ind_split_train:ind_split_val]
        test_inds = inds[ind_split_val:]
        train_set = [all_data[i] for i in train_inds]
        val_set = [all_data[i] for i in val_inds]
        test_set = [all_data[i] for i in test_inds]
        return train_set, val_set, test_set



    def generate_question_keywords_pool(self):
        key_word_pool = {k:set() for k in self.data.keys()}
        for i in range(len(self.data)):
            for k in self.data.keys():
                key_word_pool[k].add(self.data[k].iat[i])
        return key_word_pool

    def sample_wrong_answer_from_pool(self, keys, answers):
        match = True
        while match:
            new_answer = self.sample_once(keys)
            assert len(new_answer) == len(answers)
            for na,a in zip(new_answer,answers):
                if na!=a:
                    match=False
                    break
        return {k:v for k,v in zip(keys,new_answer)}

    def sample_once(self, keys):
        out = []
        for k in keys:
            candidates = self.question_keywords_pool[k]
            out.append(random.choice(tuple(candidates)))
        return out

    def remove_empty_path(self):
        self.data = self.data[self.data['path'].apply(os.path.exists)]


    def add_image_path(self):
        im_path = []
        base_path = 'images'
        for _,a in self.data.iterrows():
            folder = a['article_id'][:3]
            file_name = a['article_id'] + '.jpg'
            path = os.path.join(self.root_path,base_path, folder, file_name)
            im_path.append(path)
        self.data['path'] = im_path