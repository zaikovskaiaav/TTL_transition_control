import numpy as np
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel
from thequickmath.reduced_models.dynamical_systems import DynamicalSystem

class MoehlisFaisstEckhardtModelControl(MoehlisFaisstEckhardtModel):
    def f(self, u, action):

        f_ = np.zeros((self.dim,))
        f_[0] = self.beta ** 2 / self.Re \
                - self.beta ** 2 / self.Re * u[0] \
                - np.sqrt(3. / 2) * self.beta * self.gamma / self.k_abg * u[5] * u[7] \
                + np.sqrt(3. / 2) * self.beta * self.gamma / self.k_bg * u[1] * u[2] \
                + action[0]

        f_[1] = - (4. * self.beta ** 2 / 3. + self.gamma ** 2) / self.Re * u[1] \
                + (5 * np.sqrt(2)) / (3 * np.sqrt(3)) * self.gamma ** 2 / self.k_ag * u[3] * u[5] \
                - self.gamma ** 2 / (np.sqrt(6) * self.k_ag) * u[4] * u[6] \
                - self.alpha * self.beta * self.gamma / (np.sqrt(6) * self.k_ag * self.k_abg) * u[4] * u[7] \
                - np.sqrt(3. / 2) * self.beta * self.gamma / self.k_bg * u[0] * u[2] \
                - np.sqrt(3. / 2) * self.beta * self.gamma / self.k_bg * u[2] * u[8] \
                + action[1]

        f_[2] = - (self.beta ** 2 + self.gamma ** 2) / self.Re * u[2] \
                + (2. / np.sqrt(6.)) * self.alpha * self.beta * self.gamma / (self.k_ag * self.k_bg) * (
                            u[3] * u[6] + u[4] * u[5]) \
                + (self.beta ** 2 * (3 * self.alpha ** 2 + self.gamma ** 2) - 3 * self.gamma ** 2 * (
                    self.alpha ** 2 + self.gamma ** 2)) / (np.sqrt(6) * self.k_ag * self.k_abg * self.k_bg) * u[3] * u[
                    7] \
                + action[2]

        f_[3] = - (3 * self.alpha ** 2 + 4 * self.beta ** 2) / (3 * self.Re) * u[3] \
                - self.alpha / np.sqrt(6) * u[0] * u[4] \
                - 10. / (3. * np.sqrt(6)) * self.alpha ** 2 / self.k_ag * u[1] * u[5] \
                - np.sqrt(3. / 2) * self.alpha * self.beta * self.gamma / (self.k_ag * self.k_bg) * u[2] * u[6] \
                - np.sqrt(3. / 2) * self.alpha ** 2 * self.beta ** 2 / (self.k_abg * self.k_ag * self.k_bg) * u[2] * u[
                    7] \
                - self.alpha / np.sqrt(6.) * u[4] * u[8] \
                + action[3]

        f_[4] = - (self.alpha ** 2 + self.beta ** 2) / self.Re * u[4] \
                + self.alpha / np.sqrt(6) * u[0] * u[3] \
                + self.alpha ** 2 / (np.sqrt(6) * self.k_ag) * u[1] * u[6] \
                - self.alpha * self.beta * self.gamma / (np.sqrt(6) * self.k_abg * self.k_ag) * u[1] * u[7] \
                + self.alpha / np.sqrt(6) * u[3] * u[8] \
                + 2. / np.sqrt(6.) * self.alpha * self.beta * self.gamma / (self.k_ag * self.k_bg) * u[2] * u[5] \
                + action[4]

        f_[5] = - (3 * self.alpha ** 2 + 4 * self.beta ** 2 + 3 * self.gamma ** 2) / (3 * self.Re) * u[5] \
                + self.alpha / np.sqrt(6) * u[0] * u[6] \
                + np.sqrt(3. / 2) * self.beta * self.gamma / self.k_abg * u[0] * u[7] \
                + 10. / (3. * np.sqrt(6.)) * (self.alpha ** 2 - self.gamma ** 2) / self.k_ag * u[1] * u[3] \
                - 2. * np.sqrt(2. / 3) * self.alpha * self.beta * self.gamma / (self.k_ag * self.k_bg) * u[2] * u[4] \
                + self.alpha / np.sqrt(6) * u[6] * u[8] \
                + np.sqrt(3. / 2) * self.beta * self.gamma / self.k_abg * u[7] * u[8] \
                + action[5]

        f_[6] = - (self.alpha ** 2 + self.beta ** 2 + self.gamma ** 2) / self.Re * u[6] \
                - self.alpha / np.sqrt(6) * (u[0] * u[5] + u[5] * u[8]) \
                + 1. / np.sqrt(6) * (self.gamma ** 2 - self.alpha ** 2) / self.k_ag * u[1] * u[4] \
                + 1. / np.sqrt(6) * (self.alpha * self.beta * self.gamma) / (self.k_ag * self.k_bg) * u[2] * u[3] \
                + action[6]

        f_[7] = - (self.alpha ** 2 + self.beta ** 2 + self.gamma ** 2) / self.Re * u[7] \
                + 2. / np.sqrt(6.) * (self.alpha * self.beta * self.gamma) / (self.k_abg * self.k_ag) * u[1] * u[4] \
                + self.gamma ** 2 * (3 * self.alpha ** 2 - self.beta ** 2 + 3 * self.gamma ** 2) / (
                            np.sqrt(6) * self.k_ag * self.k_abg * self.k_bg) * u[2] * u[3] \
                + action[7]

        f_[8] = - 9 * self.beta ** 2 / self.Re * u[8] \
                + np.sqrt(3. / 2) * self.beta * self.gamma / self.k_bg * u[1] * u[2] \
                - np.sqrt(3. / 2) * self.beta * self.gamma / self.k_abg * u[5] * u[7] \
                + action[8]

        return f_


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


def rk4_timestepping(model: DynamicalSystem, ic, delta_t, n_steps, time_skip=1, space_skip=1, debug=True, stop_condition=lambda state: False):
    timeseries = np.zeros((int(n_steps//time_skip) + 1, int(model.dim//space_skip)))
    cur_state = ic
    for k in range(n_steps):
        if k % time_skip == 0:
            timeseries[int(k//time_skip), :] = cur_state[::space_skip]
        if (k % int(n_steps//10) == 0) and debug:
            print('Step {} out of {}'.format(k, n_steps))
        k_1 = delta_t*model.f(cur_state)
        k_2 = delta_t*model.f(cur_state + k_1/2.)
        k_3 = delta_t*model.f(cur_state + k_2/2.)
        k_4 = delta_t*model.f(cur_state + k_3)
        cur_state = cur_state + 1/6. * (k_1 + 2*k_2 + 2*k_3 + k_4)
        if stop_condition(cur_state):
            return timeseries[:int(k//time_skip) + 1, :]
    return timeseries