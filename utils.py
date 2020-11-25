import numpy as np

class LogisticFuncUtils(object):

    @staticmethod
    def logistic_function(x,x0=0.,k=1.,L=1.,startval=0):
        """
        return the logistic function output for the provided value
        y = L/(1 + np.exp(-k*(x-x0)))

        A useful function to use as velocity curve when interpolating velocities between two values. The derivatives are all continuous (zero jerk).
        
        :param x: value for which the logistic function is to be computed
        :type x: float
        :param x0: the x-axis value of the sigmoid's midpoint
        :type x0: float
        :param k: logistic growth rate or steepness of curve
        :type k: float
        :param L: end value of curve
        :type L: float
        :param startval: start value of curve
        :type startval: float
        :return: function output
        :rtype: float
        """
        return (L-startval)/(1 + np.exp(-k*(x-x0))) + startval

    @staticmethod
    def inverse_logistic_function(y,x0=0.,k=1.,L=1.,startval=0):

        return -np.log((L-startval)/(y-startval) - 1)/k + x0

    @staticmethod
    def solve_for_k(x, y,x0=0.,L=1,startval=0):
        return -(np.log((L-startval)/(y-startval) - 1))/(x-x0)