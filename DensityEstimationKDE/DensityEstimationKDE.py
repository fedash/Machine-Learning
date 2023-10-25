import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import statistics
from scipy.stats import pearsonr
np.random.seed(123)

# Import the data
data = pd.read_csv("data/n90pol.csv")

# -----------------------
# QUESTION 5.A
# -----------------------

n = len(data['amygdala'])
amygdala = data['amygdala']
acc = data['acc']

#1. HISTOGRAM
def choose_bins(data):
    n=len(data)

    #Sturge's rule
    bins_sturge = 1 + np.ceil(np.log2(n))
    print(f"Sturge's rule: {int(bins_sturge)}")

    #Freedman-Diaconis
    q1, q3 = np.percentile(data, [25, 75])
    width_fd = (2 * (q3-q1)) / (n ** (1 / 3))
    bins_fd = int(np.ceil((np.max(data) - np.min(data)) / width_fd))
    print(f"Freedman-Diaconis rule: {bins_fd}")

    #Square Root rule
    bins_root = int(np.sqrt(n))
    print(f"Square Root rule: {bins_root}")
    return statistics.mode([int(bins_sturge), bins_fd, bins_root])

def plot_hist(data, num_bins, var_name):
    plt.hist(data, bins=num_bins, color='lightblue', edgecolor='darkblue', linewidth=1.2)
    plt.title(f'Histogram of {var_name} with {num_bins} bins')
    plt.xlabel(var_name)
    plt.ylabel('Frequency')
    plt.show()


# ----Implementation----
#Select the number of bins for each variable
choose_bins(data=amygdala)
choose_bins(data=acc)
bins_amygdala = 8
bins_acc = 9
plot_hist(data=amygdala, num_bins=bins_amygdala, var_name = 'AMYGDALA')
plot_hist(data=acc, num_bins=bins_acc, var_name = 'ACC')
# -----------x-----------


#2. KDE
def cv_bandwidth_for_one(data, cv):
    data_vals = data.values

    # Bandwidth balues to test
    bandwidth_range = np.arange(0.001, 1, 0.001)
    bandwidths = np.round(bandwidth_range, 3)

    # Perform CV to find the best bandwidth
    cv_kde = KernelDensity(kernel='gaussian')
    bands_cv= GridSearchCV(cv_kde, {'bandwidth': bandwidths}, cv=cv)
    bands_cv.fit(data_vals.reshape(-1, 1))

    # Select the best bandwidth
    best_bandwidth = bands_cv.best_params_['bandwidth']
    print(f'Best Bandwidth: {best_bandwidth}')
    return best_bandwidth

def kde_1d(data, best_bandwidth, var_name):
    #Run KDE using the best bandwidth from cross-validation
    data_vals = data.values
    kde = KernelDensity(kernel = 'gaussian', bandwidth = best_bandwidth)
    kde.fit(data_vals.reshape(-1, 1))

    #Density estimation
    spc = np.linspace(data_vals.min(), data_vals.max(), 1000)
    est = kde.score_samples(spc.reshape(-1,1))
    density = np.exp(est)

    #Visualize the result
    plt.plot(spc, density, color='steelblue')
    plt.fill_between(spc, density, color='lightblue', alpha=0.5)
    plt.title(f'KDE for {var_name} with h={best_bandwidth}', wrap=True)
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.show()
    return density

# ----Implementation----
bandwidth_amygdala = cv_bandwidth_for_one(data=amygdala, cv=5)
kde_amygdala = kde_1d(data=amygdala, best_bandwidth=bandwidth_amygdala, var_name='AMYGDALA')
bandwidth_acc = cv_bandwidth_for_one(acc, cv=5)
kde_acc = kde_1d(data=acc, best_bandwidth=bandwidth_acc, var_name='ACC')
# -----------x-----------


# -----------------------
# QUESTION 5.B
# -----------------------

def hist_2d(data, var_name1, var_name2):
    #Square root rule to choose # of bins
    bins_2d = int(np.sqrt(len(data)))

    #Store variables
    var1 = data.iloc[:,0]
    var2 = data.iloc[:,1]

    #Plot a 2D histogram
    plt.hist2d(var1.values, var2.values, bins=bins_2d, cmap='BuPu')
    plt.xlabel(var_name1)
    plt.ylabel(var_name2)
    plt.title(f'2D Histogram with {bins_2d} bins', wrap=True)
    plt.colorbar(label='Frequency')
    plt.show()

def hist_2d_in_3d(data, var_name1, var_name2):
    #Square root rule to choose # of bins
    bins_2d = int(np.sqrt(len(data)))

    #Store variables
    var1 = data.iloc[:,0]
    var2 = data.iloc[:,1]

    #Histogram in 2D
    hist, x_var1, y_var2 = np.histogram2d(var1, var2, bins=bins_2d)

    #Project to 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xpos, ypos = np.meshgrid(((x_var1[:-1] + x_var1[1:]) / 2), ((y_var2[:-1] + y_var2[1:]) / 2))
    cmap = plt.cm.get_cmap('cool')
    set_colors = cmap(hist.ravel() / np.max(hist))
    ax.bar3d(xpos.ravel(), ypos.ravel(), np.zeros_like(hist).ravel(), np.diff(x_var1)[0], np.diff(y_var2)[0],
             hist.ravel(), color=set_colors, alpha=0.9)
    ax.set_xlabel(var_name1)
    ax.set_ylabel(var_name2)
    ax.set_zlabel('Frequency')
    ax.set_title(f'Histogram of {var_name1} & {var_name2}', wrap=True)
    plt.show()


# ----Implementation----
amyglada_acc = data.iloc[:, :-1]
hist_2d(data=amyglada_acc, var_name1='AMYGLDALA', var_name2='ACC')
hist_2d_in_3d(data=amyglada_acc, var_name1='AMYGLDALA', var_name2='ACC')
# -----------x-----------


# -----------------------
# QUESTION 5.C
# -----------------------
def cv_bandwidth_for_two(data, cv):
    var1 = data.iloc[:,0].values
    var2 = data.iloc[:,1].values

    # Bandwidth values to test
    bandwidth_range = np.arange(0.001, 1, 0.001)
    bandwidths = np.round(bandwidth_range, 3)

    # Perform CV to find the best bandwidth
    cv_kde = KernelDensity(kernel='gaussian')
    bands_cv = GridSearchCV(cv_kde, {'bandwidth': bandwidths}, cv=cv)
    bands_cv.fit(np.column_stack((var1, var2)))

    # Select the best bandwidth
    best_bandwidth = bands_cv.best_params_['bandwidth']
    print(f'Best Bandwidth: {best_bandwidth}')
    return best_bandwidth

def kde_2d(data, best_bandwidth, var_name1, var_name2, extra_var_name = None, plot3d=False):
    var1 = data.iloc[:,0].values
    var2 = data.iloc[:,1].values

    # Run KDE using the best bandwidth from CV
    kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth)
    kde.fit(np.column_stack((var1, var2)))
    x_values = np.linspace(-0.1, 0.1, 1000)
    y_values = np.linspace(-0.1, 0.1, 1000)
    xs, ys = np.meshgrid(x_values, y_values)
    xys = np.column_stack((xs.ravel(), ys.ravel()))
    est = kde.score_samples(xys)
    density = np.exp(est).reshape(1000, 1000)

    #Visualize the result: 2D
    fig, ax = plt.subplots()
    contour_range = np.linspace(density.min(), density.max(), 13)
    add_contours = ax.contourf(xs, ys, density, levels=contour_range, cmap='BuPu')
    ax.scatter(var1, var2, s=5, color='Steelblue')
    plt.colorbar(add_contours, label='Density')
    if extra_var_name == None:
        plt.title(f'KDE for {var_name1} and {var_name2} with h={best_bandwidth}', wrap=True)
    else:
        plt.title(f'KDE for {var_name1} and {var_name2}, {extra_var_name}, with h={best_bandwidth}', wrap=True)
    plt.xlabel(var_name1)
    plt.ylabel(var_name2)
    plt.show()

    #Visualize the result: 3D
    if plot3d == True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xs, ys, density, cmap='BuPu')
        ax.set_xlabel(var_name1)
        ax.set_ylabel(var_name2)
        ax.set_zlabel('Density')
        if extra_var_name == None:
            plt.title(f'KDE for {var_name1} and {var_name2} with h={best_bandwidth}', wrap=True)
        else:
            plt.title(f'KDE for {var_name1} and {var_name2}, {extra_var_name}, with h={best_bandwidth}', wrap=True)
        plt.show()

    return density


# ----Implementation----
bandwidth_acc_amygdala = cv_bandwidth_for_two(data=amyglada_acc, cv=5)
kde_amygdala_acc = kde_2d(data=amyglada_acc, best_bandwidth=bandwidth_acc_amygdala,
                   var_name1='AMYGDALA', var_name2='ACC', plot3d=True)
#Check if amygdala and acc are independent:
# 1 - Is there any linear relationship?
corr, p = pearsonr(amygdala,acc)
print(f"Pearson's correlation coefficient: {corr:.4f}")
print(f'P-value={p:.4f}')
a = 0.01
if p < a:
    print(f'Reject H0. There is no linear relationship between Amygdala and Acc at a {1-a} significance level.')
else:
    print(f'Accept H0. There is no linear relationship between Amygdala and Acc at a {1-a} significance level.')
# 2 - Based on KDE, does P(X,Y)=P(X)P(Y) hold?
p_xy = np.outer(kde_amygdala, kde_acc)
indep_kde = np.allclose(kde_amygdala_acc, p_xy, atol=0.1)
if indep_kde:
    print('Based on KDE results, Amygdala and Acc are likely independent.')
else:
    print('Based on KDE results, Amygdala and Acc are likely NOT independent. P(X,Y)=P(X)P(Y) does not hold')
# 3 - How much mutual information do the variables share?
mut_inf = mutual_info_regression(amygdala.values.reshape(-1, 1), acc.values)
print(f'Estimated mutual information shared by Amygdala and Acc: {mut_inf[0]:.4f}')
# -----------x-----------


# -----------------------
# QUESTION 5.D
# -----------------------

# ----Implementation----
#subsets for each c
data_c2 = data[data['orientation'] == 2]
data_c3 = data[data['orientation'] == 3]
data_c4 = data[data['orientation'] == 4]
data_c5 = data[data['orientation'] == 5]

#AMYGDALA (Use LOOCV since the samples are smaller now)
band_amygdala_c2 = cv_bandwidth_for_one(data=data_c2['amygdala'], cv=LeaveOneOut())
kde_1d(data=data_c2['amygdala'], best_bandwidth=band_amygdala_c2, var_name='(AMYGDALA | Orientation=2)')

band_amygdala_c3 = cv_bandwidth_for_one(data=data_c3['amygdala'], cv=LeaveOneOut())
kde_1d(data=data_c3['amygdala'], best_bandwidth=band_amygdala_c3, var_name='(AMYGDALA | Orientation=3)')

band_amygdala_c4 = cv_bandwidth_for_one(data=data_c4['amygdala'], cv=LeaveOneOut())
kde_1d(data=data_c4['amygdala'], best_bandwidth=band_amygdala_c4, var_name='(AMYGDALA | Orientation=4)')

band_amygdala_c5 = cv_bandwidth_for_one(data=data_c5['amygdala'], cv=LeaveOneOut())
kde_1d(data=data_c5['amygdala'], best_bandwidth=band_amygdala_c5, var_name='(AMYGDALA | Orientation=5)')

#ACC
band_acc_c2 = cv_bandwidth_for_one(data=data_c2['acc'], cv=LeaveOneOut())
kde_1d(data=data_c2['acc'], best_bandwidth=band_acc_c2, var_name='(ACC | Orientation=2)')

band_acc_c3 = cv_bandwidth_for_one(data=data_c3['acc'], cv=LeaveOneOut())
kde_1d(data=data_c3['acc'], best_bandwidth=band_acc_c3, var_name='(ACC | Orientation=3)')

band_acc_c4  = cv_bandwidth_for_one(data=data_c4['acc'], cv=LeaveOneOut())
kde_1d(data=data_c4['acc'], best_bandwidth=band_acc_c4, var_name='(ACC | Orientation=4)')

band_acc_c5 = cv_bandwidth_for_one(data=data_c5['acc'], cv=LeaveOneOut())
kde_1d(data=data_c5['acc'], best_bandwidth=band_acc_c5, var_name='(ACC | Orientation=5)')

#Sample means
amygdala_means = [data_c2['amygdala'].mean(), data_c3['amygdala'].mean(), data_c4['amygdala'].mean(), data_c5['amygdala'].mean()]
acc_means = [data_c2['acc'].mean(), data_c3['acc'].mean(), data_c4['acc'].mean(), data_c5['acc'].mean()]
sample_means = pd.DataFrame({'AMYGDALA': amygdala_means, 'ACC': acc_means}, index=['c=2', 'c=3', 'c=4', 'c=5'])
print(sample_means)
# -----------x-----------


# -----------------------
# QUESTION 5.E
# -----------------------

# ----Implementation----
band_acc_amygdala_c2 = cv_bandwidth_for_two(data=data_c2, cv=LeaveOneOut())
kde_amygdala_acc = kde_2d(data=data_c2, best_bandwidth=band_acc_amygdala_c2,
                   var_name1='AMYGDALA', var_name2='ACC', extra_var_name='Orientation=2', plot3d=True)
band_acc_amygdala_c3 = cv_bandwidth_for_two(data=data_c3, cv=LeaveOneOut())
kde_amygdala_acc = kde_2d(data=data_c3, best_bandwidth=band_acc_amygdala_c3,
                   var_name1='AMYGDALA', var_name2='ACC', extra_var_name='Orientation=3', plot3d=True)
band_acc_amygdala_c4 = cv_bandwidth_for_two(data=data_c4, cv=LeaveOneOut())
kde_amygdala_acc = kde_2d(data=data_c4, best_bandwidth=band_acc_amygdala_c4,
                   var_name1='AMYGDALA', var_name2='ACC', extra_var_name='Orientation=4', plot3d=True)
band_acc_amygdala_c5 = cv_bandwidth_for_two(data=data_c5, cv=LeaveOneOut())
kde_amygdala_acc = kde_2d(data=data_c5, best_bandwidth=band_acc_amygdala_c5,
                   var_name1='AMYGDALA', var_name2='ACC', extra_var_name='Orientation=5', plot3d=True)
# -----------x-----------
