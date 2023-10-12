import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english')


# print("loading......... covidqa_kg")
# with open('../Data/covidQA_kg.json','r') as f:
#     covidqa_kg=json.load(f)
# print("loaded covidqa_kg")

relation_list=['uses','treats','prevents','isa','diagnoses','co-occurs_with','associated_with','affects']

def return_kg(final_keyword,covidqa_kg):
    kg_triplet=[]
    tail=[]
    for word in final_keyword:
        if word not in nltk_stopwords and len(word) > 2 and word in covidqa_kg.keys():
            for j, triples in enumerate(covidqa_kg[word]):
                # print(triples)
                if triples[0] in relation_list:
                    kg_triplet.append([word.replace("_"," "),triples[0],triples[1].replace("_"," ")])
                    tail.append(triples[1])
    return kg_triplet, tail

def n_hops(text,kg_dic,n=2):
    NAF = ["_NAF_H", '_NAF_R', "_NAF_T"]
    triples_len = 512
    triple=[]
    tail=[]
    #1st hops
    T1,tail1 = return_kg(text,kg_dic)
    triple.extend(T1)
    tail.extend(tail1)
    if n==1:
        # print("hop==1")
        return triple
    T2,t2 = return_kg(tail1,kg_dic)
    triple.extend(T2)
    tail.extend(t2)
    # print("len of tail: ",len(tail))
    # print("len of triples: ",len(triple))
    # if len(triple)< triples_len:
    #     triple = triple + [NAF]*(triples_len - len(triple))
    # else:
    #     triple = triple[:triples_len]
    # if len(triple) == 0:
    #     triple.append([NAF]*triples_len)
    return triple


# question_text="What is the main cause of HIV-1 infection in children?"

# q= re.sub(r'[^\w\s]','',question_text)
# q_tokens=q.split()

# final_kg_triplet = n_hops(q_tokens)

