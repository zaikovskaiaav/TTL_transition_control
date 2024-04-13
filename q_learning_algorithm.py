import numpy as np
import time
import random

def count_delta(q_n, q_o, t):
    av_d = 0.0
    for i in range(len(q_n)):
        for j in range(len(q_n[0])):
            av_d += abs(q_n[i][j] - q_o[i][j])
    return av_d/(len(q_n)*len(q_n[0])*t)
    # max_d = 0.0
    # for i in range(len(q_n)):
    #     for j in range(len(q_n[0])):
    #         d = abs(q_n[i][j] - q_o[i][j])
    #         if max_d < d:
    #             max_d = d
    # return max_d


def q_learning(env, filename, time_filename, save_res, episodes=1, epsilon=0.15, alpha=0.1, gamma=0.9,
               is_rand=True, seed=None, show_real=False, show_discr=None, a_comp=None, null_action = 0):
    if a_comp is None:
        a_comp=env.model.dim
    q_table = np.loadtxt(filename)

    start_time = time.time()
    for i in range(episodes):
        actions_arr = []
        print("Iteration ", i)
        q_old = np.copy(q_table)
        state = env.reset(is_rand, seed)

        done = False
        states_traj = np.array([state])

        while not done:
            # action = null_action
            if env.time < 40:
                action = null_action
            else:
                # if env.lam:
                #     action = null_action
                    # print(env.action_space[0])
                if random.uniform(0, 1) < epsilon:
                    action = random.randrange(len(env.action_space))
                else:
                    action = np.argmax(q_table[state])

            actions_arr.append(action)

            next_state, reward, done = env.step(action)

            new_value = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
            q_table[state, action] = new_value

            state = next_state
            states_traj = np.append(states_traj, state)
        if save_res:
            # np.savetxt(f'time_series/ep_trajectories_{i}.txt', states_traj, fmt='%1u')
            np.savetxt(filename, q_table)

        print(count_delta(q_table, q_old, env.T))

        # if env.time < env.T:
        #     env.show_trajectory(discr=states_traj)

        if show_discr:
            env.show_trajectory(real=show_real, discr=states_traj)
        else:
            env.show_trajectory(real=show_real)
        # env.show_actions(actions_arr, a_comp)
        env.show_actions(actions_arr[int(20/(env.delta_t*env.n_steps)):], a_comp)

    print("Training finished.\n")
    tt = time.time() - start_time
    print(f"{tt // 60} min, {tt % 60} sec")

    if save_res:
        with open(time_filename, "a") as time_file:
            t_time = str(round(tt/60, 2))
            time_file.write(f"{t_time} {env.T} {episodes}\n")