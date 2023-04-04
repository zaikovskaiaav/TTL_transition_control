import random
import numpy as np
from s_discretization import random_initial_conditions, states_clustering, get_laminar_states
from a_discretization import get_action_space, get_action_limits, get_B
from mfe_model import MoehlisFaisstEckhardtModelControl, rk4_timestepping_control


class Environment:
    def __init__(self, action_space, m, cl, lam):
        self.action_space = action_space
        self.n_a = len(action_space)
        # self.state_space = state_space
        self.states_clusters = cl
        self.n_s = len(cl.cluster_centers)
        self.lam_states = lam

        self.model = m
        self.cur_state = np.zeros(self.model.dim)

    def reset(self):
        self.cur_state = random_initial_conditions(self.model.dim)
        print('ic:', self.cur_state)
        self.model.reset_action()
        return self.states_clusters.transform(np.array([self.cur_state]))

    def get_reward(self, state):
        return -sum((state - self.lam_states)**2)

    #
    # def get_discrete_state(self):
    #     new_state = self.states_clusters.transform(self.cur_state)
    #     return new_state

    def step(self, action, n_steps=1000):
        self.model.set_action(self.action_space[action])
        self.cur_state = rk4_timestepping_control(self.model, self.cur_state, delta_t=0.001, n_steps=n_steps)[-2]
        print(self.cur_state)
        next_state = self.states_clusters.transform(np.array([self.cur_state]))
        reward = self.get_reward(next_state)
        if reward == 0:
            return next_state, reward, True
        return next_state, reward, False



if __name__ == "__main__":

    Re = 500.0
    Lx = 1.75 * np.pi
    Lz = 1.2 * np.pi
    model = MoehlisFaisstEckhardtModelControl(Re, Lx, Lz)

    trajectory = np.loadtxt('time_series/trajectory_for_clustering.txt')
    clust_u, assign_u = states_clustering('kmeans_uniform', trajectory, n_iter_max=1000, n_cl=1000)
    lam_states = get_laminar_states(clust_u, assign_u, model)

    a_vec, actions = get_action_space(get_action_limits(get_B(model, trajectory)), 5)

    env = Environment(actions, model, clust_u, lam_states[0])


    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    all_epochs = []
    all_penalties = []

    q_table = np.zeros([env.n_s, env.n_a])

    for i in range(1, 101):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randrange(len(env.action_space))
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action, 5000)
            print(i, reward)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # if reward == -10:
            #     penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            # clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")