import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
#Function used for the model
def uncovered_area(a1, b1, r1, a2, b2, r2):
    d = math.sqrt((a2 - a1)**2 + (b2 - b1)**2)
    if d >= r1 + r2:
        return math.pi * r1**2
    if d <= abs(r2 - r1):
        return math.pi * r1**2 - math.pi * r2**2
 
    alpha = math.acos((r1**2 + d**2 - r2**2) / (2 * r1 * d))
    beta = math.acos((r2**2 + d**2 - r1**2) / (2 * r2 * d))

    area1 = 0.5 * r1**2 * (2 * alpha - math.sin(2 * alpha))
    area2 = 0.5 * r2**2 * (2 * beta - math.sin(2 * beta))
    uncovered_area = area1 + area2
    
    return math.pi * r1**2 -uncovered_area
#Background simulation to generate the model
def model(ratio,impact,t_transit):
    R1=100
    R2=ratio*R1
    v=2*(R1+R2)/t_transit
    time_span=10*R1/v
    time_sample=np.linspace(0,time_span,1000)
    normalized_time=time_sample-np.median(time_sample)
    a=0
    b=0
    a1=-5*R1
    b1=0
    flux_points2=[]
    for i in range(len(time_sample)):
        flux_points2.append(uncovered_area(a,b,R1,(a1+v*time_sample[i]),b1+(impact*R1),R2))
    flux_points2=-2.5*np.log10(np.array(flux_points2)/max(flux_points2))
    return normalized_time, flux_points2
#Fitting Function
def Fitting(data_time, data_flx, visualize=True, errordisplay=False):
    '''
    This function is used for light curve fitting.
    The input should be a time data (in days unit) for the first argument, and magnitude for the second argument
    User can choose to display error or visualize the plot using argument True or False
    '''
    def Rsquare(params):
        init_ratio, init_impact, init_transit, init_center, init_base = params
        x, y = model(init_ratio, init_impact, init_transit)
        center = init_center
        base = init_base
        yfit = np.interp(data_time, x + data_time[0] + center, y + base)
        y_mean = np.mean(data_flx)
        SST = np.sum((data_flx - y_mean)**2)
        SSR = np.sum((yfit - data_flx)**2)
        R_squared = (SST - SSR) / SST
        return -R_squared 
    if errordisplay==False: 
        bounds = [(0, 1), (0, 1), (1e-8, 1), (0, (max(data_time)-min(data_time))), (min(data_flx), max(data_flx))]   
        result = differential_evolution(Rsquare, bounds)
        result_final = minimize(Rsquare, result.x, method='BFGS')  

        best_R_squared = -result_final.fun 
        best_params = result_final.x

        print("Best R_squared:", best_R_squared)
        print("Best parameters:", best_params)

        init_ratio, init_impact, init_transit, init_center, init_base = best_params
        x, y = model(init_ratio, init_impact, init_transit)
        center = init_center
        base = init_base
        if visualize==True:
            plt.figure(figsize=(16,5))
            plt.plot(x + data_time[0] + center, y + base, color='red')
            plt.scatter(data_time, data_flx)
            plt.xlabel('time (days)')
            plt.ylabel('normalized mag')
            plt.gca().invert_yaxis()
            plt.show()

        print("rasio= ", init_ratio)
        print("impact param= ", init_impact)
        print("transit duration= ", init_transit)
    else:
        def Rsquare(params, data_time=data_time, data_flx=data_flx):
            init_ratio, init_impact, init_transit, init_center, init_base = params
            x, y = model(init_ratio, init_impact, init_transit)
            center = init_center
            base = init_base
            yfit = np.interp(data_time, x + data_time[0] + center, y + base)
            y_mean = np.mean(data_flx)
            SST = np.sum((data_flx - y_mean)**2)
            SSR = np.sum((yfit - data_flx)**2)
            R_squared = (SST - SSR) / SST
            return -R_squared  
        bounds = [(0, 1), (0, 1), (1e-8, 1), (0, (max(data_time)-min(data_time))), (min(data_flx), max(data_flx))]   
        result = differential_evolution(Rsquare, bounds)
        result_final = minimize(Rsquare, result.x, method='BFGS') 
        best_R_squared = -result_final.fun 
        best_params = result_final.x
        bootstrap_params = []
        for _ in range(100):
            indices = np.random.choice(len(data_time), size=len(data_time), replace=True)
            bootstrap_time = data_time[indices]
            bootstrap_flx = data_flx[indices]

            bootstrap_result = minimize(Rsquare, best_params, method='BFGS', args=(bootstrap_time, bootstrap_flx))
            bootstrap_params.append(bootstrap_result.x)

        bootstrap_params = np.array(bootstrap_params)
        parameter_uncertainties = np.std(bootstrap_params, axis=0)
        
        print("Best parameters:")
        print("    ratio =", best_params[0], "±", parameter_uncertainties[0])
        print("    impact param =", best_params[1], "±", parameter_uncertainties[1])
        print("    transit duration =", best_params[2], "±", parameter_uncertainties[2])

        init_ratio, init_impact, init_transit, init_center, init_base = best_params
        x, y = model(init_ratio, init_impact, init_transit)
        center = init_center
        base = init_base
        if visualize==True:
            plt.figure(figsize=(16, 5))
            plt.plot(x + data_time[0] + center, y + base, color='red')
            plt.scatter(data_time, data_flx)
            plt.xlabel('time (days)')
            plt.ylabel('normalized mag')
            plt.gca().invert_yaxis()
            plt.show()
def Lightcurve_Stacking(timedata, fluxdata, visualize=False):
    '''
    This function is used for multiple light curves Stacking.
    The data is a list consisting of another list of data, each for the time and magnitude
    User can choose to visualize the stacked data using argument True or False
    '''
    timeall=[]
    fluxall=[]
    for i in range(len(timedata)):
        center,base=np.arrange(timedata[i],fluxdata[i])
        timedata[i]=timedata[i]-(timedata[i][0]+center)
        fluxdata[i]=fluxdata[i]-base
        fluxall.extend(fluxdata[i])
        timeall.extend(timedata[i])
    pairs = list(zip(timeall, fluxall))
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    timeall = [pair[0] for pair in sorted_pairs]
    fluxall = [pair[1] for pair in sorted_pairs]
    if visualize==True:
        plt.scatter(timeall, fluxall)
        plt.gca().invert_yaxis()
        plt.show()
    return timeall, fluxall
