# Anjali - I am not a fan of the notebook look. So made this in "base" pytjon. But feel free to change it

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

from scipy.interpolate import interp1d

# data input 
M2 = pd.read_csv('jens_data/full_2018_data_M2_Bering_combined_data.csv')

M2=M2.set_index(M2.index.astype(int))
M2

# plots 
fig, ax1 = plt.subplots()
ax1.plot(M2['time_ak'], M2['full_chla'], color='green')
ax2 = ax1.twinx()
ax2.plot(M2['time_ak'], M2['full_temperature'], color='red')
plt.title('M2 chlorophyll and temp (10m)')
plt.xlabel('Time')
ax1.set_ylabel('Chlor ug/L')
ax2.set_ylabel('Temp °C')
plt.show()

# 
M2[M2['time_ak'].astype(str).str.contains('2018-08')]

# par plot # 
time = [0, M2.index[3669]]
time_eval = np.linspace(M2.index[0], M2.index[3669], 3670)

I=(pd.DataFrame((M2['approxPAR'].fillna(method='bfill')))) #define irradiance function as PAR data from M2, convert to Ein m^2 s^2
I_smoothed= I.rolling(24, min_periods=1).mean()
I_interp = scipy.interpolate.interp1d(time_eval, I_smoothed['approxPAR'], kind='linear', fill_value = 'extrapolate')

T=pd.DataFrame(M2['full_temperature'].fillna(method='bfill')) #define temp function as temp data from M2
T_smoothed= T.rolling(24, min_periods=1).mean()
T_interp = scipy.interpolate.interp1d(time_eval, T_smoothed['full_temperature'], kind='linear', fill_value = 'extrapolate')

plt.plot(I)
plt.show()


############################
### setting up constants ###
############################

# small cell photo response 
P_B_S_1 = 2.0
alpha_1 = .014
beta_1 = 0

# large cell photo response 
P_B_S_2 = 3.0
alpha_2 = .04
beta_2 = 0

# temp functions 
mu1 = 0.04 # small algae u
r1 = 0.05 # small algae b / r

mu2 = 0.07 # large algae u
r2 = 0.06 # large algae b / r

Vmax_N1  =3.244235e-08
Vmax_N2 =  8.379715e-07
Qmin_N1 = 8.650280e-08 
Qmin_N2 = 2.834475e-06
K_N1 =  9.149267e-01   
K_N2 =  3.385924e+00

phyto_mortality1 = 0.12/24
phyto_mortality2 = 0.12/24



#######################
### input functions ###
####################### 

0.001*24

# input background N
def N_in(t):
    return 0.002

# input pulse
def gaussian(t, center, amp, width): 
    return amp * np.exp(-(t-center)**2 / (2*width**2))

# defining - light term for phyto 1 (small)
def P_B1(I): 
    P_B1 = P_B_S_1*(1-(np.exp(-(alpha_1*I)/P_B_S_1)))*(np.exp((-beta_1*I)/P_B_S_1)) 
    P_frac1 = P_B1/P_B_S_1
    return P_frac1


# defining - light term for phyto 2 (large)
def P_B2(I): 
    P_B2 = P_B_S_2*(1-(np.exp(-(alpha_2*I)/P_B_S_2)))*(np.exp((-beta_2*I)/P_B_S_2)) 
    P_frac2 = P_B2/P_B_S_2
    return P_frac2

# temperature and and irrandiance growth function 
def phyto_rates_max(T, I_frac1,I_frac2): 
    phyto_growth_max1 = mu1 * np.exp(r1*T) *I_frac1 #growth as a function of temp and  irradiance
    phyto_growth_max2 = mu2 * np.exp(r2*T) *I_frac2 #growth, use minimum of dimensionless nitrogen and irradiance
    return [phyto_growth_max1, phyto_growth_max2]


#
# NP (Q)
#
def npz(t, X): #this model takes in temperature, nutrients (N), and irradiance (I)
    [N,Q1, Q2,P1, P2] = X
    Temp = T_interp(t)
    I_value = I_interp(t)
    PBI1 = P_B1(I_value) #calculate dimensionaless irradiance small algae
    PBI2 = P_B2(I_value) #calculate dimensionaless irradiance large algae
    [phyto_growth_max1, phyto_growth_max2] = phyto_rates_max(Temp, PBI1, PBI2) #calculate phyto rates based on Temperature
    dN_dt = -(Vmax_N1*(N/(N+K_N1)))*P1-(Vmax_N2*(N/(N+K_N2)))*P2 + N_in(t) # + (gaussian(t, 2209, 10, 24)) 
    dQ1_dt = Vmax_N1*(N/(N+K_N1))-phyto_growth_max1*((1-Qmin_N1/Q1))*Q1 
    dQ2_dt = Vmax_N2*(N/(N+K_N2))-phyto_growth_max2*((1-Qmin_N2/Q2))*Q2
    # phytos 
    dP1_dt = phyto_growth_max1*((1-Qmin_N1/Q1))*P1-phyto_mortality1*P1
    dP2_dt = phyto_growth_max2*((1-Qmin_N2/Q2))*P2-phyto_mortality2*P2
          
    #print(PBI)
    #print(phyto_mortality)
    return [dN_dt,dQ1_dt,dQ2_dt, dP1_dt,dP2_dt]



# initial conditions #
N_0 = 0.5
Q1_0 = 2.747626e-06
Q2_0 = 4.073799e-04
P1_0 =  4.855880e+05 #  units are off - I think. I need to check this - but it might be cells pr liter (which is not the current calue)
P2_0 = 5.231063e+04  # these are high cells per liter (so not chla - didn't hjave time to change it )

init = [N_0, Q1_0,Q2_0, P1_0,P2_0]
init

M2_model = scipy.integrate.solve_ivp(npz, y0 = init, t_span = time, t_eval=time_eval, method='Radau', dense_output=False)



###
### plots 
### 

timestamp = M2['time_ak']
plt.figure()

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12 
plt.plot(timestamp, M2_model.y[0], color='blue') #N
plt.plot(timestamp, M2_model.y[3], color='green') # small cell
plt.plot(timestamp, M2_model.y[4], color='grey') #large cell
plt.title('Simplified NP model: M2 2018')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend(['N', 'small','large'])
plt.show()


# to ug_l_ chla (input is in cells / liter ) - these are rough (stolen) conversions from my r-script. Not accurate (yet)
timestamp = M2['time_ak']
plt.figure()

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12 
plt.plot(timestamp, M2_model.y[0], color='blue') #N
plt.plot(timestamp,41.18718*(1/50)/1000000*M2_model.y[3], color='green') # small cell
plt.plot(timestamp,  669.0801*(1/50)/1000000*M2_model.y[4], color='grey') #large cell
plt.title('Simplified NP model: M2 2018')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend(['N', 'small','large'])
plt.show()


