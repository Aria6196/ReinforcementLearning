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

# Buat embedding dari seluruh pertanyaan
question_embeddings = embed_sentences(questions)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def respond(user_input):
    # Ubah input menjadi embedding
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Hitung kemiripan input terhadap semua pertanyaan dataset
    similarities = util.cos_sim(input_embedding, question_embeddings)[0]
    
    # Ambil index pertanyaan paling mirip → ini digunakan sebagai index state Q-table
    best_state_idx = similarities.argmax().item()

    # Cari index jawaban terbaik berdasarkan Q-table
    best_action_idx = np.argmax(q_table[best_state_idx])
    chosen_answer = answers[best_action_idx]

    # Jika jawaban terlalu panjang, cari alternatif
    if len(chosen_answer.split()) > 100:
        for idx in np.argsort(q_table[best_state_idx])[::-1]:
            if len(answers[idx].split()) <= 100:
                return answers[idx]
        return "I'm still learning to answer that properly. Please ask something else."

    return chosen_answer

# Loop Chat
print("Chatbot siap. Ketik 'exit' untuk keluar.")
while True:
    user_input = input("Kamu: ")
    if user_input.lower() == "exit":
        break
    print("Bot :", respond(user_input))
