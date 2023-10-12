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

from load_covidqa_dic import*
from create_kg import*
from biobert_embedding_cosine_sim import*


# pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
# pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')


def context_kg_triple(context):
	final_concept= terms_output(context)
	final_dict=dict()
	count=0
	for x in final_concept:
	    c_name=x['term']
	    for i in x['semtype']:
	        # print(count)
	        count=count+1
	        if c_name in final_dict.keys():
	            triplet = remove_duplicate_triplets(readSemanticRelations(relationpath,i,x['term'],final_concept))
	            final_dict[c_name].extend(triplet)
	            final_dict[c_name] = list(set(final_dict[c_name]))
	        else:
	            triplet = remove_duplicate_triplets(readSemanticRelations(relationpath,i,x['term'],final_concept))
	            final_dict[c_name] = triplet
	return final_dict

def sorted_triple(question,context_kg,pubmed_bert_model,pubmed_bert_tokenizer):
	q= re.sub(r'[^\w\s]','',question)
	q_concept = terms_output(q)
	q_tokens=[]
	for x in q_concept:
		q_tokens.append(x['term'])
	q_tokens= list(set(q_tokens))
	final_kg_triplet = n_hops(q_tokens,context_kg,n=1)

	sentence_pairs = [question,final_kg_triplet]
	sim_final_dict= get_bert_based_similarity(sentence_pairs, pubmed_bert_model, pubmed_bert_tokenizer)

	triple_n_sim_final_dic=dict()
	for k,t in zip(sim_final_dict.keys(),final_kg_triplet):
		triple_n_sim_final_dic[k]=t

	sorted_dict = {k: v for k, v in sorted(sim_final_dict.items(), key=lambda item: item[1], reverse=True)}

	sorted_triple_list=[]
	for k,v in sorted_dict.items():
		sorted_triple_list.append(triple_n_sim_final_dic[k])

	return sorted_triple_list



# context= "Hypertensive heart disease is the No. 1 cause of death associated with high blood pressure. It refers to a group of disorders that includes heart failure, ischemic heart disease, and left ventricular hypertrophy (excessive thickening of the heart muscle). Heart failure does not mean the heart has stopped working. Rather, it means that the heart's pumping power is weaker than normal or the heart has become less elastic. With heart failure, blood moves through the heart's pumping chambers less effectively, and pressure in the heart increases, making it harder for your heart to deliver oxygen and nutrients to your body. To compensate for reduced pumping power, the heart's chambers respond by stretching to hold more blood. This keeps the blood moving, but over time, the heart muscle walls may weaken and become unable to pump as strongly. As a result, the kidneys often respond by causing the body to retain fluid (water) and sodium. The resulting fluid buildup in the arms, legs, ankles, feet, lungs, or other organs, and is called congestive heart failure. High blood pressure may also bring on heart failure by causing left ventricular hypertrophy, a thickening of the heart muscle that results in less effective muscle relaxation between heart beats. This makes it difficult for the heart to fill with enough blood to supply the body's organs, especially during exercise, leading your body to hold onto fluids and your heart rate to increase. Symptoms of heart failure include: Shortness of breath Swelling in the feet, ankles, or abdomen Difficulty sleeping flat in bed Bloating Irregular pulse Nausea Fatigue Greater need to urinate at night High blood pressure can also cause ischemic heart disease. This means that the heart muscle isn't getting enough blood. Ischemic heart disease is usually the result of atherosclerosis or hardening of the arteries (coronary artery disease), which impedes blood flow to the heart. Symptoms of ischemic heart disease may include: Chest pain which may radiate (travel) to the arms, back, neck, or jaw Chest pain with nausea, sweating, shortness of breath, and dizziness; these associated symptoms may also occur without chest pain Irregular pulse Fatigue and weakness Any of these symptoms of ischemic heart disease warrant immediate medical evaluation. Your doctor will look for certain signs of hypertensive heart disease, including: High blood pressure Enlarged heart and irregular heartbeat Fluid in the lungs or lower extremities Unusual heart sounds Your doctor may perform tests to determine if you have hypertensive heart disease, including an electrocardiogram, echocardiogram, cardiac stress test, chest X-ray, and coronary angiogram. In order to treat hypertensive heart disease, your doctor has to treat the high blood pressure that is causing it. He or she will treat it with a variety of drugs, including diuretics, beta-blockers, ACE inhibitors, calcium channel blockers, angiotensin receptor blockers, and vasodilators. In addition, your doctor may advise you to make changes to your lifestyle, including: Diet: If heart failure is present, you should lower your daily intake of sodium to 1,500 mg or 2 g or less per day, eat foods high in fiber and potassium, limit total daily calories to lose weight if necessary, and limit intake of foods that contain refined sugar, trans fats, and cholesterol. Monitoring your weight: This involves daily recording of weight, increasing your activity level (as recommended by your doctor), resting between activities more often, and planning your activities. Avoiding tobacco products and alcohol Regular medical checkups: During follow-up visits, your doctor will make sure you are staying healthy and that your heart disease is not getting worse."
# final_dict= context_kg_triple(context)
# print("len of kg_dic: ",len(final_dict.keys()))


# question_list= ["What are the symptoms of heart failure?",
# 				# "Can high blood pressure bring on heart failure?",
# 				# "What tests are used to help diagnose hypertensive heart disease?",
# 				# "What is hypertensive heart disease?",
# 				# "What are the symptoms of ischemic heart disease?",
# 				]

# for item,question_text in enumerate(question_list):	
# 	t=sorted_triple(question_text,final_dict,pubmed_bert_model,pubmed_bert_tokenizer)
# 	print("\nTop 10 triples\n\n")
# 	print(t[:10])

