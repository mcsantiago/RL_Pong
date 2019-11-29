import random
import gym
from gym import wrappers
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


# from dqnagent_keras import DQNAgent
from dqnagent_impl import DQNAgent

def rgb2gray(rgb):
    small_frame = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return small_frame

def preprocess_frame(frame):
    """ Preprocess the frame to 80x80x1 """
    frame = frame[35:195] # Crop
    frame= frame[::2,::2, 0] # Downsample    
    frame = np.expand_dims(frame, axis=2)
    frame[frame == 144] = 0 # Erase background (background type 1)
    frame[frame == 109] = 0 # Erase background (background type 2)
    frame[frame != 0] = 1   # Everything else
    frame = frame.flatten()
    frame = np.expand_dims(frame, axis=1)
    # return frame.astype(np.float).ravel()
    return frame

if __name__ == "__main__":
    env = gym.make('Pong-v0')
    # env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
    state_size = env.observation_space
    action_size = env.action_space.n

    output_dir = 'model_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Hyperparameters
    batch_size = 32
    n_frames = 10000000
    agent = DQNAgent(state_size, action_size)

    done = False
    new_record = False
    episode = 0
    max_score = 0
    k = 4 # The paper mentions only registering every kth frame

    f = open("specs.csv", "w+")

    while True:
        state = preprocess_frame(env.reset())

        player_score = 0
        enemy_score = 0
        done = False
        
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state= preprocess_frame(next_state)

            if reward > 0: player_score += reward
            else: enemy_score -= reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state


        if player_score > max_score: 
            max_score = player_score
            new_record = True
        
        print("episode: {}    enemy_score: {}    player_score: {}    high_score: {}    epsilon: {}" # print the episode's score and agent's epsilon
        .format(episode, enemy_score, player_score, max_score, agent.epsilon))
        
        error = agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        f.write("{}, {}, {}, {}, {}".format(episode, error, player_score, max_score, agent.epsilon))
        episode += 1

        if episode % 100 == 0: # save weights every 50th episode (game)
            agent.save(output_dir + "weights_" + '{:04d}'.format(episode) + ".hdf5")

    f.close()