from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def embed_sentences(sentences):
    return model.encode(sentences, convert_to_tensor=True)

def compute_similarity(a_emb, b_emb):
    return util.cos_sim(a_emb, b_emb).item()

def get_reward(user_input, chosen_answer):
    input_emb = model.encode(user_input, convert_to_tensor=True)
    answer_emb = model.encode(chosen_answer, convert_to_tensor=True)
    similarity = util.cos_sim(input_emb, answer_emb).item()

    if similarity >= 0.8:
        return 1
    elif similarity >= 0.5:
        return 0
    else:
        return -1

