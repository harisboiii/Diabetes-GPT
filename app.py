import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
import torch

import gdown
import os
import pandas as pd

# Download the file
file_id = '1P3Nz6f3KG0m0kO_2pEfnVIhgP8Bvkl4v'
url = f'https://drive.google.com/uc?id={file_id}'
excel_file_path = os.path.join(os.path.expanduser("~"), 'medical_data.csv')

gdown.download(url, excel_file_path, quiet=False)

# Read the CSV file into a DataFrame using 'latin1' encoding
try:
    medical_df = pd.read_csv(excel_file_path, encoding='utf-8')
except UnicodeDecodeError:
    medical_df = pd.read_csv(excel_file_path, encoding='latin1')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(medical_df['Questions'])

# Load pre-trained GPT-2 model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load pre-trained Sentence Transformer model
sbert_model_name = "paraphrase-MiniLM-L6-v2"
sbert_model = SentenceTransformer(sbert_model_name)

# Function to answer medical questions using a combination of TF-IDF, LLM, and semantic similarity
def get_medical_response(question, vectorizer, X_tfidf, model, tokenizer, sbert_model, medical_df):
    # TF-IDF Cosine Similarity
    question_vector = vectorizer.transform([question])
    tfidf_similarities = cosine_similarity(question_vector, X_tfidf).flatten()

    # Find the most similar question using semantic similarity
    question_embedding = sbert_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, sbert_model.encode(medical_df['Questions'].tolist(), convert_to_tensor=True)).flatten()
    max_sim_index = similarities.argmax().item()

    # LLM response generation
    input_text = "DiBot: " + medical_df.iloc[max_sim_index]['Questions']
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id
    lm_output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=pad_token_id)
    lm_generated_response = tokenizer.decode(lm_output[0], skip_special_tokens=True)

    # Compare similarities and choose the best response
    if tfidf_similarities.max() > 0.5:
        tfidf_index = tfidf_similarities.argmax()
        return medical_df.iloc[tfidf_index]['Answers']
    else:
        return lm_generated_response

# Streamlit app
st.title("DiBot")

user_input = st.text_input("You:")
if user_input.lower() == "exit":
    st.stop()
response = get_medical_response(user_input, vectorizer, X_tfidf, model, tokenizer, sbert_model, medical_df)
st.text_area("Bot's Response:", response)
