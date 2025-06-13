import json
import numpy as np
from utils import embed_sentences
from sentence_transformers import SentenceTransformer, util

# Load Q-table dan data
q_table = np.load("q_table.npy")
with open("Data/cleaned_all_datasets_shorten.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

question_embeddings = embed_sentences(questions)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def respond(user_input):
    # encode input
    input_embedding = model.encode(user_input, convert_to_tensor=True)

    # cari pertanyaan yang paling mirip
    similarities = util.cos_sim(input_embedding, question_embeddings)[0]
    # Urutkan indeks jawaban berdasarkan similarity tinggi → rendah
    top_indices = similarities.argsort(descending=True)

    # Cari jawaban yang pendek & relevan
    for idx in top_indices:
        answer = answers[idx]
        if len(answer.split()) <= 100:  # batas panjang
            return answer

    # Jika semua jawaban terlalu panjang
    return "I'm still learning to answer that properly. Please ask something else."
    

# Chat loop
print("Chatbot siap. Ketik 'exit' untuk keluar.")
while True:
    user_input = input("Kamu: ")
    if user_input.lower() == "exit":
        break
    print("Bot :", respond(user_input))
