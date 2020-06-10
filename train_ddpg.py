import numpy as np
import gym
from osim.env import L2M2019Env
import pandas as pd

import tensorflow as tf
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, Activation, Flatten, Concatenate, Add
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import layers
from keras import backend as K

from replay_buffer import ReplayBuffer
from actor import Actor
from critic import Critic

from replay_buffer import ReplayBuffer

def obs_to_state_vector(observation):

    df = pd.DataFrame.from_dict(observation)

    obs_flat = df.to_numpy().flatten()

    state = []

    for i in range(len(obs_flat)):
        val = obs_flat[i]

        if type(val) is dict:
            val = np.array(list(val.values()))
            state = np.append(state,val)
        elif type(val) is list:
            val = np.array(val)
            state = np.append(state,val)
        else:
            if ~np.isnan(val):
                state = np.append(state,[val])

    return state

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.25, theta=.05, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def bellman(rewards, q_values, dones,gamma):

    critic_target = np.asarray(q_values)
    for i in range(q_values.shape[0]):
        if dones[i]:
            critic_target[i] = rewards[i]
        else:
            critic_target[i] = rewards[i] + gamma * q_values[i]
    return critic_target

def train(env, actor, critic, actor_noise, state_dim, action_dim):

    num_ep = 10000
    batch_size = 256
    min_buffer_size = 500
    buffer_size = 20000
    gamma = 0.99985

    Reward = []

    replay_buffer = ReplayBuffer(buffer_size=buffer_size,random_seed=1234)

    total_reward = 0

    epsilon = 1
    epsilon_decay = 0.9995

    for i in range(num_ep):

        reward_ep = 0

        print("Starting New")
        obs  = env.reset(project=True, seed=None, obs_as_dict=False)
        curr_state = np.array(obs)

        env.render()

        ep_ave_max_q  = 0

        terminal = False

        ep_length = 0

        #env.render()

        while terminal==False:

            s = curr_state

            #print("Preprocessed state",s.shape)

            #print("processed state", s)

            '''
            if np.random.rand() < epsilon:
                pred_pose = np.random.rand(19)*10-5
                action = PD_control_output(pred_pose,des_coord,des_vel)
            else:
                pred_pose = actor.predict(s)[0] +actor_noise()
                action = PD_control_output(pred_pose,des_coord,des_vel)
                #action = np.clip(action+actor_noise(), env.action_space.low, env.action_space.high)
            '''

            action = actor.predict(s)[0] +actor_noise()

            #action = PD_control_output(pred_pose,des_coord,des_vel)

            #print("Action",action)
            
            next_s, r, terminal,_ = env.step(action, obs_as_dict=False)

            curr_state = np.array(next_s)

            replay_buffer.add(s, action, r, terminal, np.array(next_s))

            #print("Preprocessed Next State",next_s)

            if replay_buffer.size() > min_buffer_size:

                print("Started Training")

                s_batch, a_batch, r_batch, t_batch, next_s_batch = replay_buffer.sample_batch(batch_size)

                q_values = critic.target_predict([next_s_batch, actor.target_predict(next_s_batch)])

                critic_target = bellman(r_batch, q_values, t_batch,gamma)

                
                critic.train_on_batch(s_batch, a_batch, critic_target)

                ep_ave_max_q += np.amax(critic.target_predict([s_batch,a_batch]))
                
                actions = actor.model.predict(s_batch)
                grads = critic.gradients(s_batch, actions)
                
                actor.train(s_batch, actions, np.array(grads).reshape((-1, action_dim)))
                
                actor.transfer_weights()
                critic.transfer_weights()

            reward_ep += r
            ep_length += 1            

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d} | Epsilon: {:.4f}'.format(int(reward_ep),i, (ep_ave_max_q / float(ep_length)), ep_length,epsilon))#,(total_reward/float(i+1))))
                #print("Reward:",reward_ep,"Episode:",i,"Max Q:",ep_ave_max_q)
                pd.DataFrame(np.array(Reward)).to_csv("./results/reward.csv")

                actor.save("./results/")
                critic.save("./results/")

                if epsilon > 0.05:
                    epsilon *= epsilon_decay

        Reward.append(reward_ep)
        total_reward += reward_ep

        if (i+1) % 100==0:
            avg_reward = 0
            for i in range(100):
                avg_reward += Reward[len(Reward)-1-i]

            print("Average of last 100 episodes",avg_reward/100.0)

            if avg_reward/100.0 > 50:
                break

    pd.DataFrame(Reward).to_csv("./results/reward.csv")

    actor.save("./results/")
    critic.save("./results/")

if __name__ == '__main__':

    model = '3D'
    difficulty = 1
    seed = None
    project = True
    obs_as_dict = False

    env = L2M2019Env(seed=seed, difficulty=difficulty, visualize=False)
    env.change_model(model=model, difficulty=difficulty, seed=seed)
    obs_dict = env.reset(project=project, seed=seed, obs_as_dict=obs_as_dict)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    lr = 0.0002
    tau = 0.001

    #print(env.action_space.high,env.action_space.low)


    #state_dim = vectorized_state.shape[0]
    #action_dim = int(vectorized_state.shape[0]/4)

    '''
    print("state",vectorized_state)
    print(state_dim,action_dim)
    '''

    actor = Actor(state_dim,action_dim,env.action_space.high,0.2 * lr, tau)
    critic = Critic(state_dim,action_dim,lr, tau)

    '''

    des_state = actor.predict(vectorized_state)[0]

    print(des_state, des_state.shape)

    torques = PD_control_output(des_state,des_coord,des_vel)

    print(torques, torques.shape, env.action_space.shape)
    '''


    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    #train(env, actor, critic, actor_noise, state_dim, action_dim)


    train(env, actor, critic, actor_noise, state_dim, action_dim)