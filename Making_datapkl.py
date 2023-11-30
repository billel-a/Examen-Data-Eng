from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer


# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)


import pickle 


with open("data.pkl", 'wb') as file:
    pickle.dump((embeddings,labels), file)