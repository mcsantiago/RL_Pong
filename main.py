import random
import gym
import numpy as np
import os


from dqnagent_keras import DQNAgent

def rgb2gray(rgb):
    small_frame = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return small_frame

def preprocess_frame(frame):
    state = rgb2gray(frame)
    state = state[::2,::2]
    state = np.expand_dims(state, axis=0)
    state = np.expand_dims(state, axis=3)
    state = state[0:1, 25:110, 0:80, 0:1]
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

    done = False
    time = 0
    episode = 0
    max_score = 0
    k = 4 # The paper mentions only registering every kth frame
    while time < n_frames:
        env.reset()

        state = None
        player_score = 0
        enemy_score = 0
        action = 0
        
        while not done:
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            if reward > 0: 
                player_score += reward
            else: 
                enemy_score -= reward

            i = 0
            while i < k - 1 and not done:
                env.render()
                next_frame, next_reward, done, _ = env.step(action)
                next_frame = preprocess_frame(next_frame)
                next_state = np.append(next_state, next_frame, axis=3)
                reward += next_reward
                if next_reward > 0: 
                    player_score += next_reward
                else: 
                    enemy_score -= next_reward
                i += 1

            if state is not None:
                action = agent.act(state)
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size) # train the agent by replaying the experiences of the episode
            state = next_state
            time += 1
        episode += 1

        if player_score > max_score: 
            max_score = player_score
        
        print("frame: {}/{}    enemy_score: {}    player_score: {}    max_score: {}    e: {:.2}" # print the episode's score and agent's epsilon
        .format(time, n_frames, enemy_score, player_score, max_score, agent.epsilon))
        
        done = False
        
        
        if episode % 50 == 0: # save weights every 50th episode (game)
            agent.save(output_dir + "weights_" + '{:04d}'.format(episode) + ".hdf5")
