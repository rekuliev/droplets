import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math


class Distribution():
    def __init__(self, amount, dissipation, d_min, d_max):
        self.amount = amount  # Amount of points to calculate
        self.dissipation = dissipation  # Energy dissipation rate
        self.d_min = d_min  # Minimal size of the droplets in distribution
        self.d_max = d_max  # Maximal size of the droplets in distribution
        self.d = np.full((amount, 1), d_min, dtype='float64')  # Distribution of the droplets' size
        self.pdf = np.zeros(self.d.shape)  # Probability density function of creation droplets
        self.arg_lap = 0  # Argument for laplace function
        self.laplace = np.zeros((4, amount))  # Laplace function values
        self.p_1 = 0  # Probability of breaking up droplet to one droplet
        self.p_2 = 0  # Probability of breaking up droplet to two droplets
        self.p_3_s = 0  # Probability of creation 2 small droplets after breaking up to 3 droplets
        self.p_3_l = 0  # Probability of creation large droplet after breaking up to 3 droplets
        self.pdf_1 = 0  # Probability density function of no droplets breaking up
        self.pdf_2 = 0  # Probability density function of breaking up to 2 droplets
        self.pdf_3 = 0  # Probability density function of breaking up to 3 droplets
        self.n_1 = 0  # Fraction of particles not subjected to breaking up
        self.n_2 = 0  # Fraction of particles subjected to breaking up to 2 particles
        self.n_3 = 0  # Fraction of particles subjected to breaking up to 3 particles
        self.new_pdf = 0  # Probability density function after iteration of droplets breaking up

    def define_d(self):
        """
        Creation of droplets' size array
        """
        pass

    def calc_pdf(self, a_0=0.624, a_1=2.68 * (10 ** (-3)), a_2=5.82 * (10 ** (-4)), k=0.613, n=1.62):
        """
        Calculate discrete probability density function of droplets' size distribution
        Parameters: calculated constants for machine of required type for calculation
        """
        pass

    def input_distribution(self):
        """
        Calculation of distribution with set parameters
        """
        pass

    def calc_arg_lap(self):
        """
        Calculation of arguments for Laplace values
        """
        pass

    def int_lap(self, t):
        """
        Definition of Laplace function
        :param t: Laplace function argument
        :return: Laplace function value
        """
        pass

    def laplace_calculate(self):
        """
        Calculation of Laplace function
        """
        pass

    def find_bound_index(self, matrix, bound, right=True):
        """
        Return index of bound size for calculation
        :param matrix: matrix where to find bound index
        :param bound: bound size for calculation
        :param right: True for the left bound, False for the right bound of calculation
        :return: index of bound
        """
        pass

    def define_probability(self):
        """
        Define probability of creation of one, two and tree droplets
        """
        pass

    def define_pdf(self):
        """
        Define probability density functions for creation of one, two and tree droplets
        """
        pass

    def find_new_pdf(self):
        """
        Calculate probability density function after iteration of breaking up
        """
        pass

    def iterate(self, n=1, plot=False):
        """
        Rewrite the droplet's distribution properties after n breaking up iterations
        n : count of iterations
        plot : if plot = True, generate plot of distribution in each iteration
        """
        pass

    def plot(self):
        """
        Make plot of droplets' size distribution
        """
        pass

    def calculation(self):
        """
        Main method for calculation of whole sequence of operations to find the distribution
        after required number of iteration
        """
        pass
