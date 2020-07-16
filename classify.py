###############################################################################
## Semester:         CS 540 Spring 202
##
## This File:        classify.py
## Author:           Andy O'Connell
## Email:            ajoconnell2@wisc.edu
## CS Login:         o-connell
##
###############################################################################
##                   fully acknowledge and credit all sources of help,
##                   other than Instructors and TAs.
##
## Persons:          N/A
##
## Online sources:   Lecture Notes and Piazza
##
###############################################################################


import os
import math
import collections 

def create_vocabulary(training_directory, cutoff):
    vocabulary = []
    d = dict()
    for root, dirs, files in os.walk(training_directory):
        for file in files:
            if file.endswith(".txt"):
                file = os.path.join(root, file)
                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip()  
                        if line in d:
                            d[line] = d[line] + 1
                        else:
                            d[line] = 1
    for key in list(d.keys()):
        if d[key] >= cutoff:
            vocabulary.append(key)

    return sorted(vocabulary)

def create_bow(vocab, filepath):
    temp_list = []
    d = {}
    none_count = 0
    temp_list_copy = []

    #Reads the selected file into a temp list
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            temp_list.append(line)
    
    #Adds the corrasponding values to each word in the dictionary
    for x in range(len(temp_list)):
        count = 0
        for y in range(len(temp_list)):
            if temp_list[x] == temp_list[y]:
                count += 1
    #Creates a copy of temp_list
    for x in temp_list:
        temp_list_copy.append(x)
    
    #Gets a count of words not included in the vocab list and puts them in the dictionary
    for x in temp_list:
        in_line = check(x, vocab)
        if not in_line:
            none_count += 1
            temp_list_copy.remove(x)
        else:
            d[x] = 0
    d[None] = none_count
    
    #Gets count of words included in the vocab list and puts them in the dictionary
    for x in temp_list_copy:
        count = 0
        for y in temp_list_copy:
            if x == y:
                count += 1
        d[x] = count
    if d[None] == 0:
        del d[None]
    return d

#Helper function to see if a word is a part of a list
def check(word, list):
    if word in list:
        return True
    else:
        return False


def load_training_data(vocab, directory):
    training_dic = {}
    t_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file = os.path.join(root, file)
                if file.split("/")[1] == 'corpus':
                    label = file.split("/")[3]
                if file.split("/")[1] == 'EasyFiles':
                    label = file.split("/")[2]
                training_dic["label"] = label    
                training_dic["bow"] = create_bow(vocab, file) 
                t_list.append(training_dic.copy())
    return t_list                

def prior(training_data, label_list):
    prior_log_dict = dict()
    #calculates the number of training documents in a class
    denominator = len(training_data)

    #calculate the number of training documents overall
    length = len(training_data)
    for x in range(length):
        label = (training_data[x]['label'])
        nom_count = 0
        for y in range(length):
            if label == training_data[y]['label']:
                nom_count += 1
        prior_math = (nom_count / denominator)
        prior_log = math.log(prior_math)
        prior_log_dict[label] = prior_log
    return prior_log_dict


def p_word_given_label(vocab, training_data, label):
    #WORKS BUT HAVE TO MAKE IT WORK WHEN NONE IS ALREADY INCLUDED
    p_word = dict()
    p_word_given_label = dict()
    x = 0
    y = 0
    total_value_count = 0
    OOV_count = 0
    count_denom = 0
    is_none = False
    vocab_copy = []
    vocab_copy = vocab.copy()
    temp_list = []

    for x in vocab_copy:
        if x != vocab_copy:
            is_none = True
    if is_none:
        vocab_copy.append(None)

    #Finds the bow that correlates to the given label
    for x in range(len(training_data)):
        if training_data[x]['label'] == label:
            label_dict = training_data[x]['bow'] 
            temp_list.append(label_dict)
   
    #Combines list of dictionary and combines the same keys
    counter = collections.Counter() 
    for d in temp_list:  
        counter.update(d) 
    p_word_temp = dict(counter) 

    for x in vocab_copy:
        if check_word_in_dict(x, p_word_temp) == True:
            p_word[x] = p_word_temp.get(x)
        elif check_word_in_dict(x, p_word_temp) == False and check_word_in_dict(x, p_word_temp) == False:
            p_word[x] = 0
   
    #Calculates the total OOV count
    for x in p_word:
        if p_word.get(x) == 0 and x != None:
            OOV_count = OOV_count + 1
        elif x == None:
            OOV_count = OOV_count + p_word.get(x)

    #Take the dictionary and run the necessary calculations
    for x in p_word:
        if p_word.get(x) != 0 and x != None:
            count_denom = count_denom + (p_word.get(x) + 1)

    #Calculates Denominator
    total_value_count = (OOV_count+1) + count_denom

    #Calculate Numerator
    for word in p_word:
        x = (p_word.get(word) + 1)
        y = x / total_value_count
        p_word_given_label[word] = math.log(y)
    return p_word_given_label


def check_word_in_dict(word, dictionary):
    if word in dictionary:
        return True
    else:
        return False 

def train(training_directory, cutoff):
    vocab= create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)

    train_dict = dict()
    train_dict['vocabulary'] = vocab

    train_dict['log prior'] = prior(training_data, ['2016', '2020'])

    train_dict['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')

    train_dict['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')

    return train_dict

def classify(model, filepath):
    classify_dict = dict()
    prob_six = 0
    prob_twenty = 0

    vocab = model['vocabulary']
    bagOfWords = create_bow(vocab, filepath)
    for word in bagOfWords:
        prob_six += (model['log p(w|y=2016)'][word] * bagOfWords[word])
        prob_twenty += (model['log p(w|y=2020)'][word] * bagOfWords[word])

    prob_six += model['log prior']['2016']
    prob_twenty += model['log prior']['2020'] 

    classify_dict['log p(y=2016|x)'] = prob_six
    classify_dict['log p(y=2020|x)'] = prob_twenty

    if prob_six > prob_twenty:
        classify_dict['predicted y'] = '2016'
    else:
        classify_dict['predicted y'] = '2020'

    return classify_dict


