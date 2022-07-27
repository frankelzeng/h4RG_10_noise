"""
This is the main file to generate the training and testing set dataframe for the classification task

There are dy*dx pixels, so that axis is from pixel 0 (lower left) to dy*dx-1 (upper right). But in this problem where we are trying to categorize weird pixel behaviors, the physical location of the pixel is not information we want to use. You can set dy and dx (it is the size of the subregion you want to examine — eventually we may want the whole array but you probably want something small for testing purposes)
RTN pixels (random telegraph noise, defect in the pixel amplifier two quantum states either filled or not, random process, see pixel 3368)
cosmic ray hits (see pixel 1), 
Hot pixels (rang of "how" hot, non-zero slope),High dark current

For each pixel: linear fit of signal vs. time, find top 1% in slope and in sum of squared residuals
Full array = 4096^2
Ideally the sum of squared residuals would be close to read noise “Ideal SSR” = sigma_read^2 * n_dof (here n_dof = ntime-2)
sigma_read should be of order ~10 ADU

Useful snippet of code:
cds_image = my_file[1].data[0,0,:,:].astype(np.int32) - my_file[1].data[0,-1,:,:].astype(np.int32)
my_file[1].data[0,:,ystart:ystart+dy,xstart:xstart+dx].reshape(ntime,dy*dx)
ntime = numpy.shape(my_file[1].data)[1]
"""

import sktime
import numpy as np
import pandas as pd
import heapq
import time
from astropy.io import fits
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# define matplotlib parameters
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['axes.labelsize'] = 35
mpl.rcParams['legend.fontsize'] = 35
mpl.rcParams['axes.titlesize'] = 35
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.labelsize'] = 35
mpl.rcParams['ytick.labelsize'] = 35
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}
from matplotlib import gridspec


def plot_TimeSeries(y, name, x= np.arange(56).reshape(-1, 1), path="./", color='black', linestyle='-', linewidth=3):
    """
    Function to plot one time series curve
    """
    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 
    ax0 = plt.subplot(gs[0])
    ax0.set_yscale("linear")
    ax0.set_xscale("linear")
    ax0.set_ylabel("Amplitude")
    ax0.set_xlabel("Time Frame")
    line, = ax0.plot(x, y, color = color, linestyle=linestyle, linewidth=linewidth)
    plt.savefig(path + name + ".pdf",dpi=300, bbox_inches='tight')
    return

def display(df):
    """
    Function to display dataframe information:
    """
    print(df.describe())
    print(df.head())
    return

def colName():
    """
    Function to add column names to dataframe, here i and j index refer to the index in my_subregion, not the whole CCD
    """
    name = ['t' + str(i) for i in range(56)]
    name.append('r2_score')
    name.append('slope')
    name.append('res_sqsum')
    name.append('i_index')
    name.append('j_index')
    return name

def testSktime():
    """
    Function to test sktime
    """
    X, y = load_arrow_head()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("X_train_type", type(X_train))
    print("y_train_type", type(y_train))
    print("X_train_head(2)", X_train.head(2).to_string())
    print("y_train[:2]", y_train[:2])
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred)) #0.8679245
    return

def createDF():
    """
    Create dataframe file from raw data
    """
    # Read and analyze pixels subregion
    path = "/fs/scratch/PCON0003/cond0007/SCA21643-noise/"
    fits_file = "20201126_95k_1p1m0p1_noise_21643_001.fits"
    my_file = fits.open(path + fits_file)
    my_subregion = my_file[1].data[0,:,0:40,0:40] # larger region, flat the abnormal pixels 4/26/2022
    # takes region with y=0 to 20 and x=20 to 50, Axes are: [0, time direction, y direction index, x direction inddex]
    print("my_subregion_shape", np.shape(my_subregion))
    noise_curve0 = my_subregion[:,0,0]

    # calculating indicators for one pixel
    y = np.array(noise_curve0)
    x = np.arange(56).reshape(-1, 1) # x here is timeframe steps which should be the same for all pixels
    model = LinearRegression().fit(x, y) 
    y_pred = model.predict(x)
    r_sq = model.score(x, y)
    res_sqsum = np.sum(np.square(y_pred - y))
    slope = model.coef_[0]
    print('coefficient of determination:', r_sq)
    print("slope", slope)
    print('residual squared sum: ', res_sqsum)
    print("noise_curve", noise_curve0)
    noise_curve0 = np.append(noise_curve0, r_sq)
    noise_curve0 = np.append(noise_curve0, slope)
    noise_curve0 = np.append(noise_curve0, res_sqsum)
    noise_curve0 = np.append(noise_curve0, 0)
    X_noise = np.append(noise_curve0, 0)
    
    # Adding other pixels to the stack
    start = time.time()
    for i in range(0, len(my_subregion[0])):
        for j in range(0, len(my_subregion[0][0])):
            noise_curve_curr = my_subregion[:,i,j]
            y = np.array(noise_curve_curr)
            x = np.arange(56).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x)
            r_sq = model.score(x, y)   
            slope = model.coef_[0]
            res_sqsum = np.sum(np.square(y_pred - y))
            noise_curve_curr = np.append(noise_curve_curr, r_sq)
            noise_curve_curr = np.append(noise_curve_curr, slope)
            noise_curve_curr = np.append(noise_curve_curr, res_sqsum)
            noise_curve_curr = np.append(noise_curve_curr, i)
            noise_curve_curr = np.append(noise_curve_curr, j)
            X_noise = np.vstack((X_noise, noise_curve_curr))
    end = time.time()
    print("Total time=", end - start)
    df_noise = pd.DataFrame(X_noise, columns=colName())
    df_noise = df_noise.iloc[1:, :] # remove the duplicated first row
    display(df_noise)
    y_noise = np.array(['1', '0', '0', '1', '2', '0', '0'])
    df_noise.to_csv(r'/users/PCON0003/osu10644/h4RG_10_noise/df_noise.csv', index = True)
    return

def read_noSort():
    """
    Read csv file and plot out pixels without sorting
    """
    df = pd.read_csv("df_noise.csv")
    df = df.iloc[:, 1:]
    print(df.head())
    for i in range(len(df)):
        y = df.iloc[i, 0:56].values.tolist()
        i_index = int(df.iloc[i]['i_index'])
        j_index = int(df.iloc[i]['j_index'])
        print('i,j=', i_index, j_index)
        path = '/users/PCON0003/osu10644/h4RG_10_noise/' + 'no_sort' +'/'
        plot_TimeSeries(y, 'rank'+str(i)+'_'+'i'+str(i_index)+'_'+'j'+str(j_index), path=path)
    return

def read_sort(attribute, n=1600, asc=True):
    """
    Read csv file and sort rows in terms of a property (e.g. slope, res_sqsum) then plot out the top n pixels
    """
    df = pd.read_csv("df_noise.csv")
    df = df.iloc[:, 1:]
    df = df.sort_values(by=attribute, ascending=asc)
    print(df.head())
    for i in range(n):
        y = df.iloc[i, 0:56].values.tolist()
        i_index = int(df.iloc[i]['i_index'])
        j_index = int(df.iloc[i]['j_index'])
        print('i,j=', i_index, j_index)
        path = '/users/PCON0003/osu10644/h4RG_10_noise/' + str(attribute) +'/'
        plot_TimeSeries(y, 'rank'+str(i)+'_'+'i'+str(i_index)+'_'+'j'+str(j_index), path=path)
    return

# main code for execution
# createDF()
read_noSort()
# read_sort('slope', asc=False)
