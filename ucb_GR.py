#####################################################################################
# Creator     : Gaurav Roy
# Date        : 19 May 2019
# Description : The code performs Upper Confidence Bound Reinforcement Learning
#               algorithm on the Ads_CTR_Optimisation.csv.
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement UCB
import math
N = len(dataset)
d = len(dataset.columns)

ads_Selected = []
num_Selection = [0] * d
sum_Reward = [0] * d
total_Reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ad_index = 0
    for i in range(0, d):
        if (num_Selection[i] > 0): 
            avg_Reward = sum_Reward[i]/num_Selection[i]
            delta = math.sqrt(1.5 * math.log(n+1) / num_Selection[i])
            upper_bound = avg_Reward + delta
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad_index = i
    
    ads_Selected.append(ad_index)
    num_Selection[ad_index] = num_Selection[ad_index] + 1
    reward = dataset.values[n, ad_index]
    sum_Reward[ad_index] = sum_Reward[ad_index] + reward
    total_Reward = total_Reward + reward
    
# Visualize the Results
plt.hist(ads_Selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.grid(True)
plt.show()