import os.path
import random
import time
import numpy as np
from s_discretization import random_initial_conditions, states_clustering, get_laminar_states, show_ek
from a_discretization import get_action_space, get_action_limits, get_B
from mfe_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control


class Environment:
    def __init__(self, action_space, m, clust_model):
        self.action_space = action_space
        self.state_space_model = clust_model
        self.n_a = len(action_space)
        self.n_s = len(clust_model.cluster_centers)

        self.model = m
        self.cur_state = np.zeros(self.model.dim)

        self.eps = 1e-5
        self.time = 0
        self.T = 3000

    def reset(self):
        self.time = 0
        self.cur_state = random_initial_conditions(self.model.dim)
        # print('ic:', self.cur_state)
        return self.state_space_model.transform(np.array([self.cur_state]))

    def get_reward(self):
        # print(state)
        return -sum((self.cur_state - self.model.laminar_state)**2)

    def lam_stop_condition(self, reward):
        return -self.eps <= reward <= self.eps

    def time_stop_condition(self):
        return self.time == self.T

    def step(self, action_num, n_steps=1000):
        self.time += 1
        if (self.time % 100) == 0:
            print("t = ", self.time)
        self.cur_state = rk4_timestepping_control(self.model, self.cur_state, self.action_space[action_num],
                                                  delta_t=0.001, n_steps=n_steps)[-2]
        next_state = self.state_space_model.transform(np.array([self.cur_state]))
        reward = self.get_reward()
        if self.time_stop_condition():
            return next_state, reward, True
        elif self.lam_stop_condition(reward):
            return next_state, reward, True
        return next_state, reward, False


def q_learning(env, filename, episodes=1, epsilon=0.15, alpha=0.1, gamma=0.9, n_steps_rk=1000):
    q_table = np.loadtxt(filename)

    start_time = time.time()
    for i in range(episodes):
        print("Iteration ", i)
        state = env.reset()
        reward = 0
        done = False
        states_traj = np.array([state])

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randrange(len(env.action_space))
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action, n_steps_rk)

            new_value = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
            q_table[state, action] = new_value

            state = next_state
            states_traj = np.append(states_traj, state)

        np.savetxt(f'time_series/ep_trajectories_{i}.txt', states_traj, fmt='%1u')
        show_ek(None, [model, clust_u, states_traj, None])
        np.savetxt(filename, q_table)

    print("Training finished.\n")
    print(f"{(time.time() - start_time) // 60} min, {(time.time() - start_time) % 60} sec")




if __name__ == "__main__":
    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi
    model = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')
    clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max=1000, n_cl=1000)

    a = 10
    a_comp = 4
    a_vec, actions = get_action_space(get_action_limits(get_B(model, trajectory)), a, num_of_a=a_comp)

    env = Environment(actions, model, clust_u)

    alpha = 0.8
    gamma = 0.9
    epsilon = 0.2

    n_steps_rk = 1000

    n_episodes = 10
    filename = f'q_tables/q_table_a_{a}_acomp_{a_comp}_rk_{n_steps_rk}.gz'  # + диапазон а

    if not os.path.exists(filename):
        np.savetxt(filename, np.zeros([env.n_s, env.n_a]))

    q_learning(env, filename, n_episodes, epsilon, alpha, gamma, n_steps_rk)

