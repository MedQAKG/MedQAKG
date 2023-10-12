## https://techblog.ezra.com/semantic-similarity-measurement-in-clinical-text-c34011e67408
import numpy as np

from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

from load_covidqa_dic import*

def get_bert_based_similarity(sentence_pairs, model, tokenizer):
    """
    computes the embeddings of each sentence and its similarity with its corresponding pair

    Args:
        sentence_pairs(dict): dictionary of lists with the similarity type as key and a list of two sentences as value
        model: the language model
        tokenizer: the tokenizer to consider for the computation
    
    Returns:
        similarities(dict): dictionary with similarity type as key and the similarity measure as value
    """
    similarities = dict()
    inputs_1 = tokenizer(sentence_pairs[0], return_tensors='pt')
    sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)

    for count,triple in enumerate(sentence_pairs[1]):
        # print(count)
        inputs_2 = tokenizer(' '.join(triple), return_tensors='pt')
        sent_2_embed = np.mean(model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)
        similarities[' '.join(triple)] = dot(sent_1_embed, sent_2_embed)/(norm(sent_1_embed)* norm(sent_2_embed))
    # for sim_type, sent_pair in sentence_pairs.items():
    #     print("inputs_1: ",sent_pair[0])
    #     print("inputs_2: ",sent_pair[1])
    #     inputs_1 = tokenizer(sent_pair[0], return_tensors='pt')
    #     inputs_2 = tokenizer(sent_pair[1], return_tensors='pt')
    #     sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
    #     sent_2_embed = np.mean(model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)
    #     similarities[sim_type] = dot(sent_1_embed, sent_2_embed)/(norm(sent_1_embed)* norm(sent_2_embed))
    return similarities


## we expect to get a high number for the first pair and a low number for the second.

# ## BioBert

# if __name__ == "__main__":
#     sentence_pairs = {'similar': ['the MRI of the abdomen is normal and without evidence of malignancy', 
#                                   'no significant abnormalities involving the abdomen is observed'], 
#                      'dissimilar': ['mild scattered paranasal sinus mucosal thickening is observed', 
#                                    'deformity of the ventral thecal sac is observed']}

#     bio_bert_model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
#     bio_bert_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
#     print(get_bert_based_similarity(sentence_pairs, bio_bert_model, bio_bert_tokenizer))

#     # output:
#     # {'similar': 0.8876422, 'dissimilar': 0.8731133}
    
# ## BlueBert
# if __name__ == "__main__":
#     sentence_pairs = {'similar': ['the MRI of the abdomen is normal and without evidence of malignancy', 
#                               'no significant abnormalities involving the abdomen is observed'], 
#                  'dissimilar': ['mild scattered paranasal sinus mucosal thickening is observed', 
#                                'deformity of the ventral thecal sac is observed']}
                                   
#     blue_bert_model = AutoModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
#     blue_bert_tokenizer = AutoTokenizer.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
#     print(get_bert_based_similarity(sentence_pairs, blue_bert_model, blue_bert_tokenizer))
#     # ouput:
#     # {'similar': 0.82826704, 'dissimilar': 0.70020705}

## pubmed_bert
if __name__ == "__main__":

    question_text="What is the main cause of HIV-1 infection in children?"
    q= re.sub(r'[^\w\s]','',question_text)
    q_tokens=q.split()
    print("prepering KG ")
    final_kg_triplet = n_hops(q_tokens)
    print("KG preparation done")
    print("len of triple: ",len(final_kg_triplet))

    sentence_pairs = [question_text,final_kg_triplet]

    # sentence_pairs = {'similar': ['the MRI of the abdomen is normal and without evidence of malignancy', 
    #                               'no significant abnormalities involving the abdomen is observed'], 
    #                 'dissimilar': ['mild scattered paranasal sinus mucosal thickening is observed', 
    #                                'deformity of the ventral thecal sac is observed']}

    pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    # print(get_bert_based_similarity(sentence_pairs, pubmed_bert_model, pubmed_bert_tokenizer))
    final_dict= get_bert_based_similarity(sentence_pairs, pubmed_bert_model, pubmed_bert_tokenizer)
    print(final_dict)
    N=10
    sorted_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)[:N]}
    print(sorted_dict)
    # output:
    # {'similar': 0.9717411, 'dissimilar': 0.96878785}