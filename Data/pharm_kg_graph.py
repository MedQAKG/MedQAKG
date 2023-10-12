import pandas as pd 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk_stopwords = stopwords.words('english')

pharmkg = pd.read_csv('../raw_PharmKG-180k.csv')

print(pharmkg.columns)
print(pharmkg.loc[pharmkg['Entity1_name'] == 'sugar', 'Entity2_name'])

keys = pharmkg['Entity1_name'].tolist()
values = pharmkg['Entity2_name'].tolist()
relation = pharmkg['relationship_type'].tolist()


print(len(keys))
print(len(values))

# print(keys[:100])

# masha = open('../mashqa_data/Answers.txt', 'r')
mashc = open('../mashqa_data/Context.txt', 'r')
# mashq = open('../mashqa_data/Questions.txt', 'r')

allsentences = []
# lines = masha.readlines()
# # count = 0
# for line in lines:
#     # if count < 4:
#         allsentences.append(re.sub(r"[^A-Za-z0-9]", " ", line.replace('\n','').lower()).strip().replace(r"  +",' ').strip().lower())
#     # count += 1
lines = mashc.readlines()
# count = 0
for line in lines:
    # if count < 4:
        allsentences.append(re.sub(r"[^A-Za-z0-9]", " ", line.replace('\n','').lower()).strip().replace(r"  +",' ').strip().lower())
    # count += 1
# lines = mashq.readlines()
# # count = 0
# for line in lines:
#     # if count < 4:
#         allsentences.append(re.sub(r"[^A-Za-z0-9]", " ", line.replace('\n','').lower()).strip().replace(r"  +",' ').strip().lower())
#     # count += 1

# print(allsentences[0])
print(len(allsentences))

subgraph_list = []

for context in allsentences:
	# print('context', context)
	con_words = set([w for w in word_tokenize(context) if w not in nltk_stopwords and len(w) > 1])
	# print('context',con_words)
	subgraph_l = []
	for word in con_words:
		subgraph = []
		if word in keys:
			tails = pharmkg.loc[pharmkg['Entity1_name'] == word, 'Entity2_name'].tolist()
			relations = pharmkg.loc[pharmkg['Entity1_name'] == word, 'relationship_type'].tolist()
			for tail, rel in zip(tails,relations):
				if word == 'radioactive':
					print('first hop triples',[word, rel, tail])
				subgraph.append([word, rel, tail])
				if tail in keys:
					s_tails = pharmkg.loc[pharmkg['Entity1_name'] == word, 'Entity2_name'].tolist()
					s_relations = pharmkg.loc[pharmkg['Entity1_name'] == word, 'relationship_type'].tolist()
					for s_tail, s_rel in zip(s_tails,s_relations):
						if s_tail in con_words:#['thyroid','radioactive','hormone','swallow','tablet','normal','level','hyperthyroidism']:
							subgraph.append([tail, s_rel, s_tail])
							# print('second hop triples',[tail, s_rel, s_tail])
			subgraph_l.append(subgraph)
	print(len(subgraph_l))
	# print('extracted subgraph', subgraph_l[:2])
	subgraph_list.append((subgraph_l))

	# break



