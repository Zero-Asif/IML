import numpy as np
import random

temperature_data = np.sin(np.linspace(0, 10, 100)) * 10 + 20  

states = range(len(temperature_data) - 1)  
actions = np.arange(15, 35, 0.5)  
q_table = np.zeros((len(states), len(actions)))  
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1  

def calculate_reward(true_temp, predicted_temp):
    return -abs(true_temp - predicted_temp)  

# Q-Learning process
for episode in range(1000):
    state = random.choice(states) 
    
    for _ in range(50):  
        
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(len(actions)))  
        else:
            action = np.argmax(q_table[state])  
        
        next_state = (state + 1) % len(states)
        true_temp = temperature_data[next_state]
        predicted_temp = actions[action]
        reward = calculate_reward(true_temp, predicted_temp)
        
        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state  

state = 0  
predictions = []
for _ in range(len(states)):
    action = np.argmax(q_table[state])  
    predictions.append(actions[action])
    state = (state + 1) % len(states)

# Compare predictions with actual data
import matplotlib.pyplot as plt

plt.plot(temperature_data, label="Actual Temperature")
plt.plot(predictions, label="Predicted Temperature", linestyle="--")
plt.legend()
plt.show()
