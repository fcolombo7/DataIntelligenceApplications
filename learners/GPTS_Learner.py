from learners.Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm


class GPTS_Learner(Learner):
    
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.eligibility = np.zeros(self.n_arms)
        self.negative_threshold = 0.2
        self.penalty = 0.8
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, normalize_y = True, n_restarts_optimizer = 9)
        
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.eligibility[arm_idx] = norm.cdf(0, self.means[arm_idx], self.sigmas[arm_idx])
        self.pulled_arms.append(self.arms[arm_idx])
        
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards#.reshape(-1,1)
        
        '''if(np.isnan(x).any()):
            raise ValueError("x contains NaN") 
        if(np.isnan(y).any()):
            raise ValueError("y contains NaN")

        if(np.isinf(x).any()):
            raise ValueError("x contains inf")
        if(np.isinf(y).any()):
            raise ValueError("y contains inf")'''
            
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
        
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        
    def pull_arm(self):
        if self.t < 10:
            return np.random.choice(self.n_arms)
        penalized_arms = np.nonzero(self.eligibility > self.negative_threshold)
        penalized_means = self.means[penalized_arms] * self.penalty
        sampled_values = np.random.normal(penalized_means, self.sigmas)
        return np.argmax(sampled_values)