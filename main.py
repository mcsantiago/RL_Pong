import random
import gym
import numpy as np
import os
from matplotlib import pyplot


from dqnagent_keras import DQNAgent

def rgb2gray(rgb):
    small_frame = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return small_frame

def preprocess_frame(state):
    state = state[35:195]
    state = rgb2gray(state)
    state = state[::2,::2]
    state[state < 100] = 1
    state[state >= 100] = 0
    # pyplot.matshow(state)
    # pyplot.show()
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=3)
    state = state.astype('b')
    return state

if __name__ == "__main__":
    env = gym.make('Pong-v0')
    state_size = env.observation_space
    action_size = env.action_space.n

    output_dir = 'model_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Hyperparameters
    batch_size = 32
    n_frames = 10000000
    agent = DQNAgent(state_size, action_size)

    time = 0
    episode = 0
    max_score = 0
    k = 4 # The paper mentions only registering every kth frame
    while time < n_frames:
        next_state = preprocess_frame(env.reset())

        state = None
        player_score = 0
        enemy_score = 0
        action = 0
        done = False
        
        while not done:
            env.render()
            next_state, state_reward, done, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            if state_reward > 0: player_score += state_reward
            else: enemy_score -= state_reward

            while next_state.shape[3] < k and not done:
                next_frame, reward, done, _ = env.step(action)
                next_state = np.append(next_state, preprocess_frame(next_frame), axis=3)
                if reward > 0: player_score += reward
                else: enemy_score -= reward
                state_reward += reward

            if state is not None and next_state.shape[3] == k and state.shape[3] == k:
                # print('state {} next_state {}'.format(state.shape, next_state.shape))
                action = agent.act(state)
                agent.remember(state, action, state_reward, next_state, done)

            state = next_state
            time += 1
        episode += 1

        if player_score > max_score: 
            max_score = player_score
        
        print("frame: {}/{}    enemy_score: {}    player_score: {}    max_score: {}    e: {:.2}" # print the episode's score and agent's epsilon
        .format(time, n_frames, enemy_score, player_score, max_score, agent.epsilon))
        
        done = False
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        
        if episode % 50 == 0: # save weights every 50th episode (game)
            agent.save(output_dir + "weights_" + '{:04d}'.format(episode) + ".hdf5")
