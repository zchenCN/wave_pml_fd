"""
Finite difference solver for 1d scalar 
wave equation with perfectly matched layer(PML)

@author: zchen
@date: 2022-05-23
@reference: https://github.com/ar4/wave_1d_fd_pml/
"""

import numpy as np 

def ricker(dt, nt, peak_time, dominant_freq):
    """Ricker wavelet with specific dominant frequency"""
    t = np.arange(-peak_time, dt * nt - peak_time, dt, dtype=np.float32)
    w = ((1.0 - 2.0 * (np.pi**2) * (dominant_freq**2) * (t**2))
        * np.exp(-(np.pi**2) * (dominant_freq**2) * (t**2)))
    return w 

class Solver:
    def __init__(self, model, dx, dt, source_x, 
                    source_time=None, pml_width=10, pad_width=10):
        self.nptx = len(model) # number of grid points
        self.dx = np.float32(dx)
        self.dt = np.float32(dt)
        self.source_x = source_x
        self.pml_width = pml_width
        self.pad_width = pad_width

        # CFL
        max_vel = max(model)
        cfl = max_vel * dt / dx 
        assert cfl < 1 
        print(f'CFL number is {cfl}')

        # source time function
        if source_time is None:
            min_vel = min(model)
            nt = int((self.nptx-1) * self.dx / (min_vel * self.dt))
            dominant_freq = 10.0
            peak_time = 1 / dominant_freq
            self.source_time = ricker(dt, nt, peak_time, dominant_freq)
        else:
            self.source_time = source_time

        # pad
        self.total_pad = pml_width + pad_width
        self.nptx_padded = self.nptx + 2 * self.total_pad 
        self.model_padded = np.pad(model, (self.total_pad, self.total_pad), 'edge')

        # wavefield
        self.prev_wavefield = np.zeros(self.nptx_padded, np.float32) # previous wavefield
        self.cur_wavefield = np.zeros(self.nptx_padded, np.float32) # current wavefield
        self.wavefield = []

        # PML
        profile = 40.0 + 60.0 * np.arange(pml_width, dtype=np.float32)
        self.sigma = np.zeros(self.nptx_padded, np.float32)
        self.sigma[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.sigma[-self.total_pad:-self.pad_width] = profile
        self.sigma[:pad_width] = self.sigma[pad_width]
        self.sigma[-self.pad_width:] = self.sigma[-self.pad_width-1]

        # auxiliary function
        #self.prev_phi = np.zeros(self.nptx_padded, np.float32)
        self.cur_phi = np.zeros(self.nptx_padded, np.float32) 


    def first_x_deriv(self, f):
        """First derivative of f with respect to x"""
        fx = np.zeros(self.nptx_padded, np.float32)
        fx[self.pad_width:-self.pad_width] = (
                                            5 * f[self.pad_width-6:-self.pad_width-6]
                                          - 72 * f[self.pad_width-5:-self.pad_width-5]
                                          + 495 * f[self.pad_width-4:-self.pad_width-4]
                                          - 2200 * f[self.pad_width-3:-self.pad_width-3]
                                          + 7425 * f[self.pad_width-2:-self.pad_width-2]
                                          - 23760 * f[self.pad_width-1:-self.pad_width-1]
                                          + 23760 * f[self.pad_width+1:-self.pad_width+1]
                                          - 7425 * f[self.pad_width+2:-self.pad_width+2]
                                          + 2200 * f[self.pad_width+3:-self.pad_width+3]
                                          - 495 * f[self.pad_width+4:-self.pad_width+4]
                                          + 72 * f[self.pad_width+5:-self.pad_width+5]
                                          - 5 * f[self.pad_width+6:-self.pad_width+6]) / (27720*self.dx)
        return fx




    def second_x_deriv(self, f):
        """Second derivative of f with respect to x"""
        fxx = np.zeros(self.nptx_padded, np.float32)
        fxx[self.pad_width:-self.pad_width] = (
                                          - 735 * f[self.pad_width-8:-self.pad_width-8]
                                          + 15360 * f[self.pad_width-7:-self.pad_width-7]
                                          -  156800 * f[self.pad_width-6:-self.pad_width-6]
                                          + 1053696 * f[self.pad_width-5:-self.pad_width-5]
                                          - 5350800 * f[self.pad_width-4:-self.pad_width-4]
                                          + 22830080 * f[self.pad_width-3:-self.pad_width-3]
                                          - 94174080 * f[self.pad_width-2:-self.pad_width-2]
                                          + 538137600 * f[self.pad_width-1:-self.pad_width-1]
                                          - 924708642 * f[self.pad_width:-self.pad_width]
                                          + 538137600 * f[self.pad_width+1:-self.pad_width+1]
                                          - 94174080 * f[self.pad_width+2:-self.pad_width+2]
                                          + 22830080 * f[self.pad_width+3:-self.pad_width+3]
                                          - 5350800 * f[self.pad_width+4:-self.pad_width+4]
                                          + 1053696 * f[self.pad_width+5:-self.pad_width+5]
                                          - 156800 * f[self.pad_width+6:-self.pad_width+6]
                                          + 15360 * f[self.pad_width+7:-self.pad_width+7]
                                          - 735 * f[self.pad_width+8:-self.pad_width+8]
                                          ) / (302702400*self.dx**2)
        return fxx

    def one_step(self, nt):
        fxx = self.second_x_deriv(self.cur_wavefield)
        fx = self.first_x_deriv(self.cur_wavefield)
        phix = self.first_x_deriv(self.cur_phi)
        next_wavefield = self.model_padded**2 * self.dt**2 * (fxx + phix) / (1 + self.sigma * self.dt / 2)\
            + (2 * self.cur_wavefield - self.prev_wavefield) / (1 + self.sigma * self.dt / 2)\
                + self.sigma * self.dt * self.prev_wavefield / (2 + self.sigma * self.dt)

        # add source 
        sx = self.source_x + self.total_pad
        next_wavefield[sx] += self.dt**2 * self.source_time[nt]

        next_phi = -self.dt * self.sigma * self.cur_phi + self.cur_phi \
            - self.sigma * self.dt * fx 

        return next_wavefield, next_phi

    def step(self):
        for nt in range(len(self.source_time)):
            next_wavefield, next_phi = self.one_step(nt)
            self.cur_wavefield, self.prev_wavefield = next_wavefield, self.cur_wavefield 
            self.cur_phi, self.prev_phi = next_phi, self.cur_phi 
            self.wavefield.append(next_wavefield[self.total_pad:-self.total_pad])


if __name__ == '__main__':
    dx = 10
    dt = 0.001
    model = np.ones(101) * 1500.
    sol = Solver(model, dx, dt, 50)
    sol.step()
