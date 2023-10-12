import json


# def parse_triplet(kg_triplets,top_k):
# 	text=""
# 	head=[]
# 	if len(kg_triplets)-top_k>0:
# 		for item in kg_triplets[:top_k]:
# 			if item[0] in head:
# 				text=text+" [R] "+item[1]+" [O] "+item[2]+""
# 			else:
# 				text=text+"[S] "+item[0]+" [R] "+item[1]+" [O] "+item[2]+""
# 				head.append(item[0])
# 		return text
# 	else:
# 		num=top_k-len(kg_triplets)
# 		for item in kg_triplets[:len(kg_triplets)]:
# 			if item[0] in head:
# 				text=text+" [R] "+item[1]+" [O] "+item[2]+""
# 			else:
# 				text=text+"[S] "+item[0]+" [R] "+item[1]+" [O] "+item[2]+""
# 				head.append(item[0])
# 		for item in range(num):
# 			if '_NAF_H' in head:
# 				text=text+" [R] _NAF_R [O] _NAF_O"
# 			else:
# 				text=text+" [S] _NAF_H [R] _NAF_R [O] _NAF_O"
# 				head.append('_NAF_H')

# 		return text


# def parse_triplet(kg_triplets):
#     text=""
#     head=[]
#     # if len(kg_triplets)-top_k>0:
#     for item in kg_triplets:
#         if item[0] in head:
#             text=text+" "+item[1]+" "+item[2]+""
#         else:
#             text=text+" "+item[0]+" "+item[1]+" "+item[2]+""
#             head.append(item[0])
#     if len(text)== 0:
#         return "a"
#     return text



# def parse_triplet(kg_triplets,top_k):
#     text=""
#     head=[]
#     if len(kg_triplets)-top_k>0:
#         for item in kg_triplets[:top_k]:
#             if item[0] in head:
#                 text=text+" "+item[1]+" "+item[2]+""
#             else:
#                 text=text+" "+item[0]+" "+item[1]+" "+item[2]+""
#                 head.append(item[0])
#         return text
#     else:
#         num=top_k-len(kg_triplets)
#         for item in kg_triplets[:len(kg_triplets)]:
#             if item[0] in head:
#                 text=text+" "+item[1]+" "+item[2]+""
#             else:
#                 text=text+" "+item[0]+" "+item[1]+" "+item[2]+""
#                 head.append(item[0])
#         for item in range(num):
#             if '_NAF_H' in head:
#                 text=text+" _NAF_R _NAF_O"
#             else:
#                 text=text+"_NAF_H _NAF_R _NAF_O"
#                 head.append('_NAF_H')
#             return text

# def parse_triplet(kg_triplets,top_k):
#     text=""
#     head=[]
#     if len(kg_triplets)-top_k>0:
#         for item in kg_triplets[:top_k]:
#             if item[0] in head:
#                 text=text+" "+item[1]+" "+item[2]+""
#             else:
#                 text=text+" "+item[0]+" "+item[1]+" "+item[2]+""
#                 head.append(item[0])
#         return text
#     else:
#         num=top_k-len(kg_triplets)
#         for item in kg_triplets[:len(kg_triplets)]:
#             if item[0] in head:
#                 text=text+" "+item[1]+" "+item[2]+""
#             else:
#                 text=text+" "+item[0]+" "+item[1]+" "+item[2]+""
#                 head.append(item[0])
#         return text
#     if len(text)== 0:
#         return "_NAF_H _NAF_R _NAF_O"

def parse_triplet(kg_triplets,top_k):
    text=""
    head=[]
    if len(kg_triplets)-top_k>0:
        for item in kg_triplets[-top_k:]:
            if item[0] in head:
                text=text+" "+item[1]+" "+item[2]+""
            else:
                text=text+item[0]+" "+item[1]+" "+item[2]+""
                head.append(item[0])
        return text
    else:
        num=top_k-len(kg_triplets)
        for item in kg_triplets[-len(kg_triplets):]:
            if item[0] in head:
                text=text+" "+item[1]+" "+item[2]+""
            else:
                text=text+item[0]+" "+item[1]+" "+item[2]+""
                head.append(item[0])
        for item in range(num):
            if '_NAF_H' in head:
                text=text+" _NAF_R _NAF_O"
            else:
                text=text+" _NAF_H _NAF_R _NAF_O"
                head.append('_NAF_H')
        return text


kg_triplets = [
    [
        "virus",
        "co-occurs_with",
        "infectious"
    ],
    [
        "virus",
        "associated_with",
        "strain"
    ],
    [
        "virus",
        "co-occurs_with",
        "minor"
    ],
    [
        "virus",
        "co-occurs_with",
        "acute disease"
    ],
    [
        "identity",
        "co-occurs_with",
        "strain"
    ],
    [
        "virus",
        "associated_with",
        "acute disease"
    ],
    [
        "virus",
        "associated_with",
        "reported"
    ],
    [
        "virus",
        "co-occurs_with",
        "bronchitis"
    ],
    [
        "virus",
        "associated_with",
        "other"
    ],
    [
        "virus",
        "associated_with",
        "field"
    ],
    [
        "virus",
        "associated_with",
        "bronchitis"
    ],
    [
        "virus",
        "associated_with",
        "minor"
    ],
    [
        "virus",
        "associated_with",
        "overlap"
    ],
    [
        "virus",
        "affects",
        "strain"
    ],
    [
        "virus",
        "associated_with",
        "infectious"
    ],
    [
        "identity",
        "affects",
        "virus"
    ],
    [
        "identity",
        "co-occurs_with",
        "like"
    ],
    [
        "identity",
        "affects",
        "bronchitis"
    ],
    [
        "virus",
        "affects",
        "like"
    ],
    [
        "virus",
        "associated_with",
        "reports"
    ],
    [
        "identity",
        "affects",
        "other"
    ],
    [
        "virus",
        "affects",
        "identities"
    ],
    [
        "identity",
        "affects",
        "minor"
    ],
    [
        "identity",
        "affects",
        "acute disease"
    ],
    [
        "identity",
        "affects",
        "infectious"
    ],
    [
        "virus",
        "affects",
        "acute disease"
    ],
    [
        "virus",
        "affects",
        "identity"
    ],
    [
        "identity",
        "affects",
        "strain"
    ],
    [
        "virus",
        "associated_with",
        "report"
    ],
    [
        "identity",
        "co-occurs_with",
        "identities"
    ],
    [
        "identity",
        "affects",
        "like"
    ],
    [
        "virus",
        "affects",
        "minor"
    ],
    [
        "virus",
        "affects",
        "infectious"
    ],
    [
        "virus",
        "affects",
        "bronchitis"
    ],
    [
        "identity",
        "affects",
        "identities"
    ]
]
# kg_triplets =[]

print(parse_triplet(kg_triplets,50))
# print(type(parse_triplet(kg_triplets,100)))