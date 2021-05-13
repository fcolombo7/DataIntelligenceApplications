from learners.Learner import *
from scipy.stats import norm

class GTS_Learner(Learner):
    
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms)*1e3
        
        self.ineligibility = np.zeros(self.n_arms)
        self.negative_threshold = 0.2
        self.penalty = 0.8
        
    def pull_arm(self):
        if self.t < 10:
            return np.random.choice(self.n_arms)
        penalized_means = self.means
        penalized_means[self.ineligibility > self.negative_threshold] = \
                penalized_means[self.ineligibility > self.negative_threshold] * self.penalty
        sampled_values = np.random.normal(penalized_means, self.sigmas)    
        idx = np.argmax(sampled_values)
        return idx
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        
        self.ineligibility[pulled_arm] = norm.cdf(0, self.means[pulled_arm], self.sigmas[pulled_arm])

        n_samples = len(self.rewards_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples