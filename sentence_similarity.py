import numpy as np
import torch

import flash
from flash.text import TextClassificationData, TextEmbedder
from sentence_transformers import util
predict_data=["I like to watch tv","Watching Television is my favorite time pass","The cat was running after the butterfly","It is so windy today"]
# Wrapping the prediction data inside a datamodule
datamodule = TextClassificationData.from_lists(
    predict_data=predict_data,
    batch_size=4,
)

# We are loading a pre-trained SentenceEmbedder
model = TextEmbedder(backbone="sentence-transformers/all-MiniLM-L6-v2")

trainer = flash.Trainer(gpus = 1)

#Since this task is tackled unsupervised, the predict method generates sentence embeddings using the prediction input
embeddings = trainer.predict(model, datamodule=datamodule)

for i in range(0,len(predict_data),2):
  embed1,embed2=embeddings[0][i],embeddings[0][i+1]
  cosine_scores = util.cos_sim(embed1, embed2)
  if cosine_scores>=0.5:
      label="Similar"
  else:
      label="Not Similar"
  print("sentence 1:{} | sentence 2:{}| prediction: {}".format(predict_data[i],predict_data[i+1],label))
