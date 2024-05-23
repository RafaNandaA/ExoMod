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
    It doesn't matter if the input is in JD or other format, as long as it's unit is day, the fitting process will calculate the center point
    and the baseline for the magnitude
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
    return x + data_time[0] + center, y + base
def FittingErr(data_time, data_flx, data_flx_err, visualize=True, errordisplay=False):
    '''
    This function is the same as the previous function, but accommodate the error value for the magnitude data.
    The input of the error value can be a numpy array or just a single value (float).
    '''

    def Rsquare(params, data_time=data_time, data_flx=data_flx, data_flx_err=data_flx_err):
        init_ratio, init_impact, init_transit, init_center, init_base = params
        x, y = model(init_ratio, init_impact, init_transit)
        center = init_center
        base = init_base
        yfit = np.interp(data_time, x + data_time[0] + center, y + base)
        y_mean = np.mean(data_flx)
        SST = np.sum((data_flx - y_mean)**2)
        SSR = np.sum(((yfit - data_flx) / data_flx_err)**2)
        R_squared = 1 - (SSR / SST)
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
        # plt.scatter(data_time, data_flx)
        plt.errorbar(data_time, data_flx, yerr=data_flx_err, fmt='o', capsize=3,zorder=1)
        plt.plot(x + data_time[0] + center, y + base, color='red',zorder=2)
        plt.xlabel('time (days)')
        plt.ylabel('normalized mag')
        plt.gca().invert_yaxis()
        plt.show()
    return x + data_time[0] + center, y + base
def Lightcurve_Stacking(timedata, fluxdata, visualize=False):
    '''
    This function is used for multiple light curves Stacking.
    The data is a list consisting of another list of data, each for the time and magnitude
    User can choose to visualize the stacked data using argument True or False
    '''
    def arrange(data_time, data_flx):
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
            return -R_squared  # Negative to maximize R_squared

        bounds = [(0, 1), (0, 1), (0, 1), (0, (max(data_time)-min(data_time))), (min(data_flx), max(data_flx))]  # Define bounds for each parameter

        # Perform global optimization using Differential Evolution
        result = differential_evolution(Rsquare, bounds)

        best_R_squared = -result.fun
        best_params = result.x

        print("Best R_squared:", best_R_squared)
        print("Best parameters:", best_params)

        init_ratio, init_impact, init_transit, init_center, init_base = best_params
        return init_center, init_base
    timeall=[]
    fluxall=[]
    for i in range(len(timedata)):
        center,base=arrange(timedata[i],fluxdata[i])
        timedata[i]=timedata[i]-(timedata[i][0]+center)
        fluxdata[i]=fluxdata[i]-base
        fluxall.extend(fluxdata[i])
        timeall.extend(timedata[i])
    pairs = list(zip(timeall, fluxall))
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    timeall = [pair[0] for pair in sorted_pairs]
    fluxall = [pair[1] for pair in sorted_pairs]
    if visualize==True:
        plt.figure(figsize=(16, 5))
        plt.scatter(timeall, fluxall)
        plt.gca().invert_yaxis()
        plt.show()
    return timeall, fluxall
#Additional Feature
def residual(JD_all, mag_all, timemodel, magmodel):
    x,y=timemodel, magmodel
    mag_model=np.interp(JD_all,x,y)
    plt.figure(figsize=(16, 5))
    plt.scatter(np.array(JD_all),np.array(mag_all)-mag_model)
    plt.axhline(y=0, color='red', label='Model')
    plt.xlabel('time (days)')
    plt.ylabel('Residual mag')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
def model_LD_points(ratio,impact,t_transit, u, Swift=False):
    def num(r,u):
        """
        Linear surface brightness function for Limb Darkening Model 
        """
        return 10*(1 - u*(1-(1-r**2)**0.5))
    def generate_points(a, b, r, num_points, u):
        points = []
        for i in range(num_points):
            r_sampled = r*i/num_points
            try:
                density=int(num(i/num_points,u)*r_sampled)
            except:
                print(r_sampled)
            theta = np.linspace(-np.pi/2, 3*np.pi/2,density)
            for j in range(density):
                x = a + r_sampled * np.cos(theta[j])
                y = b + r_sampled * np.sin(theta[j])
                points.append((x, y))
        return points
    def uncovered_area_ld(a2, b2, r2, points):
        count = 0
        for point in points:
            x, y = point
            if (x-a2)**2 + (y-b2)**2 > r2**2:
                count += 1
        return count
    R1=100
    R2=ratio*R1
    v=2*(R1+R2)/t_transit
    time_span=10*R1/v
    if Swift==True:
        step=100
    else:
        step=1000
    time_sample=np.linspace(0,time_span,step)
    normalized_time=time_sample-np.median(time_sample)
    a1=-5*R1
    b1=0
    flux_points2=[]
    points=generate_points(0,0,R1,100,u)
    for i in range(len(time_sample)):
        flux_points2.append(uncovered_area_ld((a1+v*time_sample[i]),b1+(impact*R1),R2,points))
    flux_points2=-2.5*np.log10(np.array(flux_points2)/len(points))
    return normalized_time, flux_points2
def model_LD_field(ratio,impact,t_transit, u):
    def num(r, u):
        """
        Linear surface brightness function for Limb Darkening Model 
        """
        if r <= 1:
            return 10 * (1 - u * (1 - np.sqrt(1 - r**2)))
        else:
            return 0  
    def generate_field(a, b, r, num_points, u, field_size, a1, b1, ratio):
        r1=r*ratio
        field = np.zeros((field_size, field_size))  
        x = np.linspace(a - r, a + r, field_size)   
        y = np.linspace(b - r, b + r, field_size)   
        xx, yy = np.meshgrid(x, y)                  
        for i in range(field_size):
            for j in range(field_size):
                radius = np.sqrt((xx[i, j] - a)**2 + (yy[i, j] - b)**2)
                out = (xx[i, j] - a1)**2 + (yy[i, j] - b1)**2 -(r1)**2
                density = num(radius/r, u)
                if out<0:
                    density = 0
                field[i, j] = density
        return field, xx, yy
    def sum_outside_circle(field, xx, yy, a1, b1, r1):
        distances_squared = (xx - a1)**2 + (yy - b1)**2
        outside_circle_mask = distances_squared > r1**2
        field_sum = np.sum(field[outside_circle_mask])
        
        return field_sum
    R1=100
    R2=ratio*R1
    v=2*(R1+R2)/t_transit
    time_span=10*R1/v
    time_sample=np.linspace(0,time_span,500)
    normalized_time=time_sample-np.median(time_sample)
    a1=-5*R1
    b1=0
    flux_points2=[]
    field, xx, yy=generate_field(0, 0, R1, 100, u, 500, 0, 0, 0)
    for i in range(len(time_sample)):
        flux_points2.append(sum_outside_circle(field,xx,yy,a1+v*time_sample[i], b1, R2))
    flux_points=-2.5*np.log10(np.array(flux_points2)/max(flux_points2))
    return normalized_time, flux_points
