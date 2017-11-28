"""
Example code to calculate principal components using 
pre-calculated eigenvectors for bladder DVHs

IMPORTANT!
dvh_cum_arr must be a numpy array where rows are DVH volumes 
sampled every 0.1 Gy between 0 and 80 Gy, i.e.[0.0,0.1,...,79.9,80.0]
It is important to normalise the DVH volumes to a maximum of 1.0 
The resulting array wil have 801 columns and the number of rows will be the 
number of patients
"""
import numpy as np
# example of a random number array with 801 columns to represent the 
# required 0.1 Gy sampling and 0-80Gy range of the DVHs 
# 100 patient data correspond to the 100 rows 
dvh_cum_arr=np.random.random_sample((100, 801)) #replace this with patient DVHs
# load mean DVH
mean_dvh=np.load('mean_dvh.npy')
# load eigenvectors
eigenvectors=np.load('eigenvectors.npy')
# calculate principal components
# each row is a patient and each column represents the 
# principal component
pcs=np.dot(dvh_cum_arr-mean_dvh,eigenvectors)

# e.g principal component 1 is the zero-index column
pc1 = pcs[:,0]
# e.g principal component 4 is the 3rd index column
pc4 = pcs[:,3]
# e.g PC4 of the 50th patient
pc4_50th_patient=pcs[49,3]