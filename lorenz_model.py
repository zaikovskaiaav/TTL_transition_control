from thequickmath.reduced_models.lorenz import LorenzModel
from thequickmath.reduced_models.dynamical_systems import DynamicalSystem
import numpy as np

class LorenzModelControl(LorenzModel):
    def f(self, u, action):
        f_ = np.zeros(3)
        f_[0] = self.Pr * (u[1] - u[0]) + action[0]
        f_[1] = self.Ra * u[0] - u[0] * u[2] - u[1] + action[1]
        f_[2] = u[0] * u[1] - self.beta * u[2] + action[2]
        return f_

# class LorenzModelControl(LorenzModel):
#     def f(self, u, action):
#         f_ = np.zeros(3)
#         f_[0] = self.Pr * (u[1]+ action[1] - u[0])
#         f_[1] = self.Ra * u[0] - u[0] * u[2] - (u[1] + action[1])
#         f_[2] = u[0] * (u[1]+ action[1]) - self.beta * u[2]
#         return f_

# class LorenzModelControl(LorenzModel):
#     def f(self, u, action):
#         f_ = np.zeros(3)
#         f_[0] = self.Pr * (u[1] - u[0])
#         f_[1] = self.Ra * u[0] - u[0] * ( u[2] + action[2]) - u[1]
#         f_[2] = u[0] * u[1] - self.beta *( u[2] + action[2])
#         return f_

def rk4_timestepping_control(model: DynamicalSystem, ic, action, delta_t, n_steps, time_skip=1, space_skip=1, debug=True, stop_condition=lambda state: False):
    timeseries = np.zeros((int(n_steps//time_skip) + 1, int(model.dim//space_skip)))
    cur_state = ic
    a = action
    for k in range(n_steps):
        if k % time_skip == 0:
            timeseries[int(k//time_skip), :] = cur_state[::space_skip]
        # if (k % int(n_steps//10) == 0) and debug:
            # print('Step {} out of {}'.format(k, n_steps))
        k_1 = delta_t*model.f(cur_state, a)
        k_2 = delta_t*model.f(cur_state + k_1/2., a)
        k_3 = delta_t*model.f(cur_state + k_2/2., a)
        k_4 = delta_t*model.f(cur_state + k_3, a)
        cur_state = cur_state + 1/6. * (k_1 + 2*k_2 + 2*k_3 + k_4)
        # if stop_condition(cur_state):
            # return timeseries[:int(k//time_skip) + 1, :]
    return timeseries