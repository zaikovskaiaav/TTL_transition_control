import numpy as np
from thequickmath.reduced_models.models import DynamicalSystem

# TODO: новый класс для модели с возможностью управления, измененная (?) функция Рунге-Кутты.
class MoehlisFaisstEckhardtModelControl(DynamicalSystem):
    def __init__(self, Re, L_x, L_z):
        """
        This class sets up the shear-flow model from Moehlis et al. 2004. Namely, it is the right-hand side of the
        amplitude equations

        :param Re: Reynolds number
        :param L_x: domain wavelength in the x-direction
        :param L_y: domain wavelength in the y-direction
        """

        self.Re = float(Re)
        self.L_x = float(L_x)
        self.L_z = float(L_z)
        self.alpha = 2.*np.pi/L_x
        self.beta = np.pi/2.
        self.gamma = 2.*np.pi/L_z
        self.N_8 = 2.*np.sqrt(2) / np.sqrt((self.alpha**2 + self.gamma**2) * (4.*self.alpha**2 + 4.*self.gamma**2 + np.pi**2))
        self.k_ag = np.sqrt(self.alpha**2 + self.gamma**2)
        self.k_bg = np.sqrt(self.beta**2 + self.gamma**2)
        self.k_abg = np.sqrt(self.alpha**2 + self.beta**2 + self.gamma**2)

        super().__init__(9)

        self.action = np.zeros(self.dim)

    def f(self, u):
        """
        In the paper, amplitudes, denoted a_1, a_2, etc., are enumerated from 1 to 9. Here they are denoted u[0], u[1],
        etc. and enumerated from 0 to 8.
        """
        f_ = np.zeros((self.dim,))
        f_[0] =   self.beta**2/self.Re \
                - self.beta**2/self.Re * u[0]\
                - np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[5] * u[7] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_bg * u[1] * u[2] \
                + self.action[0]

        f_[1] = - (4.*self.beta**2/3. + self.gamma**2)/self.Re * u[1] \
                + (5*np.sqrt(2))/(3*np.sqrt(3))*self.gamma**2/self.k_ag * u[3] * u[5] \
                - self.gamma**2/(np.sqrt(6)*self.k_ag) * u[4] * u[6] \
                - self.alpha*self.beta*self.gamma/(np.sqrt(6)*self.k_ag*self.k_abg) * u[4] * u[7] \
                - np.sqrt(3./2)*self.beta*self.gamma/self.k_bg * u[0] * u[2] \
                - np.sqrt(3./2)*self.beta*self.gamma/self.k_bg * u[2] * u[8] \
                + self.action[1]

        f_[2] = - (self.beta**2 + self.gamma**2)/self.Re * u[2] \
                + (2./np.sqrt(6.)) * self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * (u[3]*u[6] + u[4]*u[5]) \
                + (self.beta**2*(3*self.alpha**2 + self.gamma**2) - 3*self.gamma**2*(self.alpha**2 + self.gamma**2))/(np.sqrt(6)*self.k_ag*self.k_abg*self.k_bg) * u[3]*u[7] \
                + self.action[2]

        f_[3] = - (3*self.alpha**2 + 4*self.beta**2)/(3*self.Re) * u[3] \
                - self.alpha/np.sqrt(6) * u[0]*u[4] \
                - 10./(3.*np.sqrt(6)) * self.alpha**2/self.k_ag * u[1]*u[5] \
                - np.sqrt(3./2) * self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * u[2]*u[6] \
                - np.sqrt(3./2) * self.alpha**2*self.beta**2/(self.k_abg*self.k_ag*self.k_bg) * u[2]*u[7] \
                - self.alpha/np.sqrt(6.) * u[4]*u[8] \
                + self.action[3]

        f_[4] = - (self.alpha**2 + self.beta**2)/self.Re * u[4] \
                + self.alpha/np.sqrt(6) * u[0]*u[3] \
                + self.alpha**2/(np.sqrt(6)*self.k_ag) * u[1]*u[6] \
                - self.alpha*self.beta*self.gamma/(np.sqrt(6)*self.k_abg*self.k_ag) * u[1]*u[7] \
                + self.alpha/np.sqrt(6) * u[3]*u[8] \
                + 2./np.sqrt(6.) * self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * u[2]*u[5] \
                + self.action[4]

        f_[5] = - (3*self.alpha**2 + 4*self.beta**2 + 3*self.gamma**2)/(3*self.Re) * u[5] \
                + self.alpha/np.sqrt(6) * u[0]*u[6] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[0]*u[7] \
                + 10./(3.*np.sqrt(6.)) * (self.alpha**2 - self.gamma**2)/self.k_ag * u[1]*u[3] \
                - 2.*np.sqrt(2./3)*self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * u[2]*u[4] \
                + self.alpha/np.sqrt(6) * u[6]*u[8] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[7]*u[8] \
                + self.action[5]

        f_[6] = - (self.alpha**2 + self.beta**2 + self.gamma**2)/self.Re * u[6] \
                - self.alpha/np.sqrt(6) * (u[0]*u[5] + u[5]*u[8]) \
                + 1./np.sqrt(6) * (self.gamma**2-self.alpha**2)/self.k_ag * u[1]*u[4] \
                + 1./np.sqrt(6) * (self.alpha*self.beta*self.gamma)/(self.k_ag*self.k_bg) * u[2]*u[3] \
                + self.action[6]

        f_[7] = - (self.alpha**2 + self.beta**2 + self.gamma**2)/self.Re * u[7] \
                + 2./np.sqrt(6.) * (self.alpha*self.beta*self.gamma)/(self.k_abg*self.k_ag) * u[1]*u[4] \
                + self.gamma**2*(3*self.alpha**2 - self.beta**2 + 3*self.gamma**2)/(np.sqrt(6)*self.k_ag*self.k_abg*self.k_bg) * u[2]*u[3] \
                + self.action[7]

        f_[8] = - 9*self.beta**2/self.Re * u[8] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_bg * u[1]*u[2] \
                - np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[5]*u[7] \
                + self.action[8]

        return f_

    def modes(self, x, y, z):
        """
        In the paper, modes are enumerated from 1 to 9. Here from 0 to 8.

        :return:
        """
        u = np.zeros((self.dim, 3))
        cos_pi_y = np.cos(np.pi * self.gamma / 2.)
        sin_pi_y = np.sin(np.pi * self.gamma / 2.)
        cos_g_z = np.cos(self.gamma * z)
        sin_g_z = np.sin(self.gamma * z)
        cos_a_x = np.cos(self.alpha * x)
        sin_a_x = np.sin(self.alpha * x)
        u[0, :] = np.array([np.sqrt(2)*sin_pi_y, 0, 0])
        u[1, :] = np.array([4./np.sqrt(3)*cos_pi_y**2 * cos_g_z, 0, 0])
        u[2, :] = 2./np.sqrt(4.*self.gamma**2 + np.pi**2) * np.array([0, 2.*self.gamma*cos_pi_y * cos_g_z, np.pi * sin_pi_y * sin_g_z])
        u[3, :] = np.array([0, 0, 4./np.sqrt(3)*cos_a_x * cos_pi_y**2])
        u[4, :] = np.array([0, 0, 2.*sin_a_x * sin_pi_y])
        u[5, :] = 4.* np.sqrt(2)/np.sqrt(3.*(self.alpha**2 + self.gamma**2)) * np.array([-self.gamma*cos_a_x*cos_pi_y**2*sin_g_z, 0, self.alpha*sin_a_x*cos_pi_y**2*cos_g_z])
        u[6, :] = 2.* np.sqrt(2)/np.sqrt(self.alpha**2 + self.gamma**2) * np.array([self.gamma*sin_a_x*sin_pi_y*sin_g_z, 0, self.alpha*cos_a_x*sin_pi_y*cos_g_z])
        u[7, :] = self.N_8 * np.array([np.pi*sin_a_x*sin_pi_y*sin_g_z, 2.*(self.alpha**2 + self.gamma**2)*cos_a_x*cos_pi_y*sin_g_z, -np.pi*self.gamma*cos_a_x*sin_pi_y*cos_g_z])
        u[8, :] = np.array([np.sqrt(2)*np.sin(3.*np.pi*y/2.), 0, 0])
        return u

    def kinetic_energy(self, u):
        axis = 0
        if len(u.shape) == 2:
            axis = 1
        return (2.*np.pi)**2 / (self.alpha * self.gamma) * np.sum(u**2, axis=axis)

    def three_dim_flow_field(self, a, x, y, z):
        u = self.modes(x, y, z)
        return np.sum(np.dot(a, u))

    def set_action(self, a):
        self.action = a

    def reset_action(self):
        self.action = np.zeros(self.dim)


def rk4_timestepping_control(model: DynamicalSystem, ic, delta_t, n_steps, time_skip=1, space_skip=1, debug=True, stop_condition=lambda state: False):
    timeseries = np.zeros((int(n_steps//time_skip) + 1, int(model.dim//space_skip)))
    cur_state = ic
    for k in range(n_steps):
        if k % time_skip == 0:
            timeseries[int(k//time_skip), :] = cur_state[::space_skip]
        # if (k % int(n_steps//10) == 0) and debug:
            # print('Step {} out of {}'.format(k, n_steps))
        k_1 = delta_t*model.f(cur_state)
        k_2 = delta_t*model.f(cur_state + k_1/2.)
        k_3 = delta_t*model.f(cur_state + k_2/2.)
        k_4 = delta_t*model.f(cur_state + k_3)
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