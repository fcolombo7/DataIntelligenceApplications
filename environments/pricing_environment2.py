import numpy
import numpy as np


class PricingEnvironment:
    def __init__(self, n_arms, conversion_rates, cost_per_click, n_clicks, tau):
        self.n_arms = n_arms
        self.conversion_rates = conversion_rates
        self.cpc = cost_per_click
        self.n_clicks = n_clicks
        # TODO: TAU È UN VETTORE DI VALORI ATTESI DI BINOMIAL
        self.tau = tau

    def round(self, pulled_arm):
        outcome = np.random.binomial(1, self.conversion_rates[pulled_arm])
        future_purchases = 0
        if outcome != 0:
            p = self.tau[pulled_arm]/30
            future_purchases = np.random.binomial(30, p)
        return [outcome, future_purchases, self.cpc]

    def day_round(self, pulled_arm):
        daily_rew = None
        for _ in range(self.n_clicks):
            if daily_rew is None:
                daily_rew = self.round(pulled_arm)
            else:
                daily_rew = numpy.vstack((daily_rew, self.round(pulled_arm)))
        return daily_rew
