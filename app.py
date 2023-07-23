from flask import Flask, jsonify, request, make_response
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd 

app=Flask(__name__)

# @app.route('/<string:text1>/<string:text2>')
@app.route('/post_json',methods=['POST'])
def hello_world():
  
  data1=request.get_json()
  text1=data1['text1'] 
  text2=data1['text2']
# Load pre-trained BERT model and tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
# Function to get sentence embeddings using BERT model
  def get_sentence_embedding(sentence):
      inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
      with torch.no_grad():
          outputs = model(**inputs)
      return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
# Function to calculate cosine similarity between two sentence embeddings

  def cosine_similarity_score(embedding1, embedding2):
      similarity = cosine_similarity([embedding1], [embedding2])
      return similarity[0][0]

  embedding1 = get_sentence_embedding(text1)
  embedding2 = get_sentence_embedding(text2)
  similarity_score = cosine_similarity_score(embedding1, embedding2)

  result={
     "similarity_score":str(similarity_score)
    }
  return jsonify(result)

if __name__=="__main__":
    app.run(debug=True)
    
    
    
