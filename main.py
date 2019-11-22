import random
import gym
from gym import wrappers
import numpy as np
import os
from time import time


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
    return frame
    # return frame.astype(np.float).ravel()

if __name__ == "__main__":
    # Macros
    UP_ACTION = 2
    DOWN_ACTION = 3

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

    while True:
        env.reset()

        state = None
        prev_state = None
        player_score = 0
        enemy_score = 0
        action = UP_ACTION
        done = False
        
        while not done:
            env.render()
            state, reward, done, _ = env.step(action)
            state= preprocess_frame(state)
            # state = state - prev_state if prev_state is not None else np.zeros(6400)
            # prev_state = state
            # state = np.expand_dims(state, axis=1).T
            
            if reward > 0: player_score += reward
            else: enemy_score -= reward

            i = 0
            while i < k - 1 and not done:
                env.render()
                next_frame, next_reward, done, _ = env.step(action)
                next_frame = preprocess_frame(next_frame)
                state = np.append(state, next_frame, axis=2)
                # state = state + next_frame
                reward += next_reward
                if next_reward > 0: player_score += next_reward
                else: enemy_score -= next_reward
                i += 1

            if state.shape[2] == 4: # exclude incomplete states
                action = agent.act(state)
                agent.remember(state, action, reward, done)
        episode += 1

        if player_score > max_score: 
            max_score = player_score
            new_record = True
        
        print("episode: {}    enemy_score: {}    player_score: {}    high_score: {}    epsilon: {}" # print the episode's score and agent's epsilon
        .format(episode, enemy_score, player_score, max_score, agent.epsilon))
        
        # agent.replay((int)(player_score + 1)) # train the agent by replaying the experiences of the episode
        agent.forget() # clear memory vector

        if episode % 100 == 0: # save weights every 50th episode (game)
            agent.save(output_dir + "weights_" + '{:04d}'.format(episode) + ".hdf5")
