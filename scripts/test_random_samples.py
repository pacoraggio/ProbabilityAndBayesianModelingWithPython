# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 05:47:07 2021

@author: pacor
"""





import numpy as np
from scipy.stats import norm
import seaborn as sns

path = r'C:/Users/pacor/Documents/Notebooks/Python/ProbabilityAndStatistics/Probability and Bayesian Modeling/data'
nn = np.load(path + '/sample_1000.npy')

import matplotlib.pyplot as plt
plt.hist(nn, bins = 13)



norm_1 = norm(loc = 25, scale = 6)
norm_2 = norm(loc = 100, scale = 14)

n_samples = 2000

k = (norm_1.rvs(n_samples ) + norm_2.rvs(n_samples ))/2
sns.histplot(k,
            bins=13,
            stat = 'density')


from scipy import stats

class comb_distribution(stats.rv_continuous):
    norm_1 = norm(loc = 25, scale = 6)
    norm_2 = norm(loc = 100, scale = 14)
    
    def _pdf(self, x):
        return ((norm_1.pdf(x) + norm_2.pdf(x))/2)


def draw_sample(n_samples):
    distribution = comb_distribution(a = 0, b = 150)    
    return(np.array([np.int(distribution.rvs()) for _ in np.arange(n_samples)]))


from timeit import default_timer as timer

start = timer()

np.random.seed(1234)
k = 5
n_samples = 20

l_5 = np.zeros(k)
for i in np.arange(k):
    a1 = draw_sample(n_samples)
    l_5[i] = a1.mean()
end = timer()
print(end - start)

len(l_5)

####
##  Time evaluation

start = timer()
draw_sample(n_samples).mean()
end = timer()
print(end - start)

np.random.seed(1234)
k = 100
n_samples = 20

## -- with list comprehension
start = timer()

c = np.array([draw_sample(n_samples).mean() for _ in np.arange(k)])

end = timer()
print(end - start)

## -- with for loop

np.random.seed(1234)
start = timer()

l_5 = np.zeros(k)
for i in np.arange(k):
    a1 = draw_sample(n_samples)
    l_5[i] = a1.mean()

end = timer()
print(end - start)

sns.histplot(l_5,
            bins=10,
            stat = 'density')


import timeit
import_module = "import random"
testcode = ''' 
from scipy import stats
import numpy as np
from scipy.stats import norm

class comb_distribution(stats.rv_continuous):
    norm_1 = norm(loc = 25, scale = 6)
    norm_2 = norm(loc = 100, scale = 14)
    
    def _pdf(self, x):
        return ((norm_1.pdf(x) + norm_2.pdf(x))/2)


def draw_sample(n_samples):
    distribution = comb_distribution(a = 0, b = 150)    
    return(np.array([np.int(distribution.rvs()) for _ in np.arange(n_samples)]))


def test(): 
    return(1+2)
'''
print(timeit.repeat(stmt=testcode))

import timeit
testcode = ''' 
import random
def test(): 
    return random.randint(10, 100)
'''
print(timeit.repeat(stmt=testcode))


