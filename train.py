import json
from utils import embed_sentences, get_reward
from model import QLearningChatbot
import numpy as np
import random

with open("Data/cleaned_all_datasets_shorten.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

question_embeddings = embed_sentences(questions)
answer_embeddings = embed_sentences(answers)

agent = QLearningChatbot(num_states=len(questions), num_actions=len(answers))

episodes = 1000
for episode in range(episodes):
    state_idx = random.randint(0, len(questions) - 1)
    state_question = questions[state_idx]

    action_idx = agent.choose_action(state_idx)
    chosen_answer = answers[action_idx]

    reward = get_reward(state_question, chosen_answer)
    agent.update_q_value(state_idx, action_idx, reward)

print("Training finished. Saving Q-table.")
np.save("q_table.npy", agent.q_table)
