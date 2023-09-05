import numpy as np
from scipy.stats import norm
import math



class Black:
    def __init__(self, params):
        self.params = params

    def d_plus(self, forward , k, expiry , vol):
        return ((math.log(forward / k) + 0.5 * vol * vol * expiry)
                / (vol * math.sqrt(expiry)))

    def d_minus(self, forward , k, expiry , vol):
        return (self.d_plus(forward = forward , k = k, expiry = expiry ,vol = vol) - vol * math.sqrt(expiry))

    def discount(self, rate, t, val):
        return math.exp(-rate*t)*val

    def price_cap(self, forward, k, expiry, vol, rate, t, isCall):
        option_value = 0
        #Check weather option is call or put
        if expiry * vol == 0.0:
            if isCall:
                option_value = math.max(forward - k, 0.0)
            else:
                option_value = math.max(k - forward, 0.0)
        else:
            d1 = self.d_plus(forward, k, expiry, vol)
            d2 = self.d_minus(forward, k , expiry, vol)
            if(isCall):
                option_value = forward * norm.cdf(d1) - k * norm.cdf(d2)
            else:
                option_value = k * norm.cdf(-d2) - forward * norm.cdf(-d1)

        return self.discount(rate, t, option_value)


