import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math


class InputDistribution:
    def __init__(self, amount, dissipation, d_min, d_max):
        self.amount = amount  # Amount of points to calculate
        self.dissipation = dissipation  # Energy dissipation rate
        self.d_min = d_min  # Minimal size of the droplets in distribution
        self.d_max = d_max  # Maximal size of the droplets in distribution
        self.d = np.full((amount, 1), d_min, dtype='float64')  # Droplets' size for calculation
        self.pdf = np.zeros(self.d.shape)  # Probability density function of droplets' size

    def define_d(self):
        """
        Creation of droplets' size array
        :return: droplets' size for calculation
        """
        return self.d + (((self.d_max - self.d_min) / self.amount) * np.arange(self.d.size).reshape(self.d.shape))

    def calc_pdf(self, a_0=0.624, a_1=2.68 * (10 ** (-3)), a_2=5.82 * (10 ** (-4)), k=0.613, n=1.62):
        """
        Calculate discrete probability density function of droplets' size distribution
        Parameters: calculated constants for required machine type
        :return: Probability density function of droplets' size
        """
        return ((a_0 + a_1*self.d + a_2*(self.d**2))/(self.d**5)) * np.exp(k - n/self.d)


class Laplace:
    def __init__(self, distribution: InputDistribution):
        self.input_distribution = distribution  # input distribution
        self.arg_lap = self.calc_arg_lap()  # Argument for Laplace function
        self.laplace = self.calculate_laplace()  # Laplace function values

    def calc_arg_lap(self):
        """
        Calculation of arguments for Laplace function
        :return: arguments for Laplace function
        ind=0: no breaking up
        ind=1: right boundary of creation 2 droplets with current size
        ind=2: left boundary of creation 2 droplets with current size
        """
        return np.array(
            [
                (3/((2 ** 0.5) * (self.input_distribution.dissipation ** (1/3)) * np.power(self.input_distribution.d, 5/6))),
                (3.82/((2 ** 0.5) * (self.input_distribution.dissipation ** (1/3)) * np.power((self.input_distribution.d * pow(2, 1/3)), 5/6))),
                (3/((2 ** 0.5) * (self.input_distribution.dissipation ** (1/3)) * np.power((self.input_distribution.d * pow(2, 1/3)), 5/6))),
                (3.82/((2 ** 0.5) * (self.input_distribution.dissipation ** (1/3)) * np.power(self.input_distribution.d, 5/6)))
            ]
        )

    @staticmethod
    def int_lap(t):
        """
        Definition of Laplace function
        :param t: Laplace function argument
        :return: Laplace function value
        """
        return np.exp(-(t ** 2))

    def calculate_laplace(self):
        """
        Calculation of Laplace function
        :return: Laplace function values
        """
        laplace = [0] * len(self.arg_lap)
        for i in range(len(self.arg_lap)):
            laplace[i] = [integrate.quad(self.int_lap, 0, b)[0] for b in self.arg_lap[i]]
        laplace = np.array(laplace)
        return laplace


class DistributionCalculation:
    def __init__(self, distribution: InputDistribution):
        self.laplace = Laplace(distribution)
        self.input_distribution = distribution
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
        Calculate probability density function after an iteration of breaking up
        """
        pass


class Distribution:
    def __init__(self, n=1):
        self.distribution = DistributionCalculation()
        self.input_distribution = self.distribution.input_distribution
        self.n = n

    def calculate(self):
        """
        Main method for calculation of whole sequence of operations to find the distribution
        after required number of iteration
        """
        pass

    def plot(self):
        """
        Make plot of droplets' size distribution
        """
        pass