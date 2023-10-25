import matplotlib.pyplot as plt
import numpy as np
np.random.seed(555)
#Distribution parameters
mu_0, sigma_0, mu_1, sigma_1, n_0, n_1 = 0, 1, 0.5, 1.5, 100, 100

#Generate
points_0 = np.random.normal(mu_0, sigma_0, n_0)
points_1 = np.random.normal(mu_1, sigma_1, n_1)
points = np.concatenate([points_0, points_1])

#PDFs
pdf_0 = np.exp(-(points - mu_0)**2 / (2 * sigma_0**2)) / (np.sqrt(2 * np.pi) * sigma_0)
pdf_1 = np.exp(-(points - mu_1)**2 / (2 * sigma_1**2)) / (np.sqrt(2 * np.pi) * sigma_1)

#log-likelihood ratio
ll = np.log(pdf_1/pdf_0)

#CUSUM
#Initiate - zeros like in the formula
cusum = np.zeros_like(ll)
#iterate through moments of time
for t in range(1, len(cusum)):
    cusum[t] = max(0, cusum[t-1] + ll[t])

#Visualize
plt.plot(cusum)
plt.title('CUSUM Statistic')
plt.ylabel('CUSUM')
plt.xlabel('Time $t$')
plt.axvline(x=100, color='r', linestyle='--')
plt.show()

plt.plot(points)
plt.title('Sequential data \n $f_0=N(0, 1), f_1=N(0.5, 1.5)$')
plt.xlabel('Time $t$')
plt.ylabel('$Samples$')
plt.axvline(x=100, color='r', linestyle='--')
plt.show()