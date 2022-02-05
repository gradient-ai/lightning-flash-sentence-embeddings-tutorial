from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

#two columns with indexes corresponding to pairs
col1=["I like to watch tv","The cat was running after the butterfly"]
col2=["Watching Television is my favorite time pass","It is so windy today"]


#Compute encodings for both lists
vectors1 = model.encode(col1, convert_to_tensor=True)
vectors2 = model.encode(col2, convert_to_tensor=True)

#Computing the cosine similarity for every pair
cosine_scores = util.cos_sim(vectors1, vectors2)

#Display cosine similarity score for the computed embeddings


for i,(sent1,sent2) in enumerate(zip(col1,col2)):
    if cosine_scores[i][i]>=0.5:
      label="Similar"
    else:
      label="Not Similar"
    print("sentence 1:{} | sentence 2:{}| prediction: {}".format(sent1,sent2,label))

    