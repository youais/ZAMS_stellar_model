import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define constants
a = 7.56577e-15 # erg cm^-3 K^-4
NA = 6.022142e23 # mol^-1
k = 1.380650e-16 # erg K^-1
G = 6.6743e-8 # cm^3 g^-1 s^-2
c = 3e10 #cm s^-1
sigma_sb = 5.6703744e-5 #g s^-3 K^-4
M_sun = 1.99e33 # g
R_sun = 6.96e10 # cm
L_sun = 3.847e33 # erg s^-1
T_eff_sun = 5780 # K
rho_avg_sun = 1.41 # g cm^-3

# Define stellar values
M_star = 1.0*M_sun
X = 0.7 #0.8 #0.7
Y = 0.27 #0.1 #0.27
Z = 1-X-Y
mu = np.divide(4, (3+5*X)) # Fully ionised, so ion mean molecular weight
Gamma2 = 5/3 # Assuming fully ionised ideal gas
del_ad = np.divide(Gamma2-1, Gamma2) # HKT 3.94

# Calculate some reasonable guesses for the stellar variables, given M_star
# Using homology relations
R_star = np.power(np.divide(M_star, M_sun), 0.75)*R_sun # HKT 1.87
L_star = np.power(np.divide(M_star, M_sun), 3.5)*L_sun # HKT 1.88
# Using constant density stellar model
P_c = np.divide(3*G*np.square(M_star), 8*np.pi*np.power(R_star, 4)) # HKT 1.42
T_c = 1.15e7*mu*np.divide(M_star, M_sun)*np.divide(R_sun, R_star) # HKT 1.56
guesses = np.asarray([P_c, R_star, L_star, T_c])
print(f'Guesses:\n    P_c = {P_c:.2e} dyne cm^-2\n    R_star = {R_star:.2e} cm\n    L_star = {L_star:.2e} erg s^-1\n    T_c = {T_c:.2e} K')

# Define fractions of enclosed mass to set as integration bounds
M_fracs = np.asarray([1e-12, 0.9999, 0.5])

# Converge model
converged_sol = converger(guesses, M_star=M_star, M_fracs=M_fracs, verbose=False)

# MESA stellar parameters
# For M=1.33M_sun, Z=0.03
Rmesa = np.power(10, 0.117989)
Lmesa = np.power(10, 0.401320)
Tmesa = np.power(10, 7.226664)

# Calculate % error off MESA
# Pressure
print(f'{converged_sol.x[0]:.2e} dyne cm^-2')
# Radius
print(f'{converged_sol.x[1]/R_sun:.2e} R_sun, {abs(1-((converged_sol.x[1]/R_sun)/Rmesa))*100:.3}% of MESA')
# Luminosity
print(f'{converged_sol.x[2]/L_sun:.2e} L_sun, {abs(1-((converged_sol.x[2]/L_sun)/Lmesa))*100:.3}% of MESA')
# Temperature
print(f'{converged_sol.x[3]:.2e} K, {abs(1-(converged_sol.x[3]/Tmesa))*100:.3}% of MESA')

# Get final values
sol = findSolution(converged_sol.x, M_star, M_fracs)

# Get pressure, radius, luminosity, and temperature profiles
masses = sol[0]/M_star
pressures = sol[1]/converged_sol.x[0]
radii = sol[2]/converged_sol.x[1]
luminosities = sol[3]/converged_sol.x[2]
temperatures = sol[4]/converged_sol.x[3]

# Plot pressure, radius, luminosity, and temperature profiles
plt.figure(figsize=(10, 7))
plt.plot(masses, pressures, label='P')
plt.plot(masses, radii, label='R')
plt.plot(masses, luminosities, label='L')
plt.plot(masses, temperatures, label='T')
plt.legend()
plt.show()

# Plot energy generation rates as in KWW fig 18.8
logTs = np.linspace(6, 8, 100)
ePPs = []
eCNOs = []
eHs = []
for i in np.arange(len(logTs)):
    pp, CNO, eH = findEGR(T=10**logTs[i], rho=1.41)
    ePPs.append(pp)
    eCNOs.append(CNO)
    eHs.append(eH)
plt.figure(figsize=(7, 7))
plt.plot(logTs, np.log10(ePPs), label='pp-chain', linestyle='--', color='cornflowerblue')
plt.plot(logTs, np.log10(eCNOs), label='CNO cycle', linestyle='--', color='limegreen')
plt.plot(logTs, np.log10(eHs), label='total ε', linestyle='--', color='black')
plt.ylim(-10, 10)
plt.xlabel('log$T$')
plt.ylabel('log$ε_H$') #[$erg$ $g^{-1}$ $s^{-1}$]')
plt.title('Stellar energy generation rate')
plt.legend()
plt.show()
