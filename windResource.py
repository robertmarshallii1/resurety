import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


def main():
    ## Load power curve data
    power_curve_data = pd.read_csv('PowerCurve.csv')

    ## Load met tower data
    met_data = pd.read_csv('wind.csv')
    met_data.Time = pd.to_datetime(met_data.Time)
    wind_data = met_data[met_data['WindSpeed60.ms'] >= 0].iloc[:,0:3] # Remove bad wind data

    ## Compute wind shear coefficient from 30-m and 60-m wind data
    wind_data['alpha'] = (np.log(wind_data['WindSpeed60.ms']) - np.log(wind_data['WindSpeed30.ms']))/(np.log(60) - np.log(30))

    ## Calculate 80-m wind speed using power law 
    wind_data['ws80'] = wind_data['WindSpeed60.ms']*(80/60)**wind_data.alpha
    wind_data.loc[wind_data.ws80.isnull(),['ws80']] = 0 # Alpha undefined where wind is zero due to ln(0) in numerator. Replace nan values with zero. 

    ## Average 80-m wind speed for 2011
    avg80 = np.mean(wind_data.ws80)
    print(f'Average 80-m wind speed in 2011: {avg80:.2f} m/s\n')

    ## Monthly average 80-m wind speeds
    month_avg = dict()
    for m in wind_data.Time.dt.month_name().unique():
        month_avg[m] = np.mean(wind_data[wind_data.Time.dt.month_name() == m].ws80)
    for m in month_avg:
        print(f'Average 80-m wind speed in {m} 2011: {month_avg[m]:.2f} m/s')

    ## September 2011 energy production estimate
    power = monthlyEnergy(power_curve_data,wind_data,9)
    print(f'\nSeptember 2011 energy production estimate: {power:.2f} MWh')

    ## Plot data
    register_matplotlib_converters()
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax[0].plot(wind_data.Time,wind_data['WindSpeed30.ms'],'.')
    ax[0].plot(wind_data.Time,wind_data['WindSpeed60.ms'],'.')
    ax[0].plot(wind_data.Time,wind_data.ws80,'.')
    ax[0].legend(['30m','60m','80m'],loc=1)
    ax[0].set(ylabel='Wind Speed (m/s)')
    ax[1].plot(met_data.Time,met_data['Dir.degree'],'.')
    ax[1].set(ylabel='Wind Direction ($^\circ$)')
    ax[2].plot(met_data.Time,met_data['Temp.C'],'.')
    ax[2].set(ylabel='Air Temperature ($^\circ$C)')
    ax[2].axhline(y=0,linestyle=':',color='k')
    plt.show()

def monthlyEnergy(power_curve,wind_data,month):
    ## Returns an estimate of monthly energy production from 10 turbines, in MWh, for a given month
    month_data = wind_data[wind_data.Time.dt.month == month].ws80 # Grab data points in specified month
    wind_bins = np.arange(-0.125,30.126,0.25) # Define bins for wind speed
    f = np.histogram(month_data, bins=wind_bins)[0] # Get frequency counts for each wind speed bin 
    f = f/np.sum(f) # Normalized frequencies
    f = f*np.array(wind_data[wind_data.index == month_data.index[0]].Time.dt.daysinmonth)[0]*24 # Expected hours spent in each bin for the month
    return np.sum(f*power_curve['Power.kW'])*10/1000 # Return estimate of monthly energy production from 10 turbines in MWh

if __name__=="__main__":
    main()