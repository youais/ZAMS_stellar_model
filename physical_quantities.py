import numpy as np
import pandas as pd

from scipy.interpolate import griddata

# Read in OPAL opacity table
opaltb = pd.read_csv('./opal_table74.txt', sep=' ', header=2, index_col=0)
opaltb.columns = opaltb.columns.astype('float64')

# Set up points and values for opacity interpolation routine
points = []
values = []
for i in range(len(opaltb.columns.values)):
    for j in range(len(opaltb.index.values)):
        points.append((opaltb.columns.values[i], opaltb.index.values[j]))
        values.append(opaltb.values[j][i])
points = np.array(points, dtype='float64')
values = np.array(values, dtype='float64')

# Set up interpolation routine
def findOpacity(LogT, Logrho=None, LogR=None, method='linear', checkOPAL=False, verbose=False):
    '''
    A function to calculate stellar opacity.

    Acquires density and temperature values, along with corresponding
    opacity values, from OPAL table. Then uses scipy.interpolate.griddata
    to interpolate opacity values across OPAL opacity grid given an
    input density and temperature.
    '''
    #Logrho = np.log10(rho) # Should work between -9 and 3
    #LogR = np.log10(rho/np.power(T*1e-6, 3))
    #LogT = np.log10(T) # Should work between 3.75 and 7.5
    if checkOPAL:
        ogLogkappa = opaltb[LogR].loc[LogT]
        Logrho = LogR + 3*np.log10((10**LogT)*1e-6)
    else:
        ogLogkappa = 'N/A'
        LogR = Logrho - 3*np.log10((10**LogT)*1e-6)
    Logkappa = griddata(points, values, (LogR, LogT), method=method)
    kappa = 10**Logkappa
    if verbose:
        print(f'Interpolating opacity...\n   Inputting log(ρ)={Logrho:.3f}, log(T)={LogT:.3f}, log(R)={LogR:.3f} \n   Outputting log(κ) values...\n      ',
                     f'From OPAL table: {ogLogkappa}\n      ',
                     f'From linear interpolator: {Logkappa}')#\n   ',
                     #f'Nearest interpolator: {nearLogkappa}\n   ',
                     #f'Cubic interpolator: {cubeLogkappa}')
        if ogLogkappa == Logkappa and checkOPAL:
            print('Yay! Interpolator running correctly :)')
    return kappa

# Set up energy generation rate calculation
def findEGR(T, rho, verbose=False):
    '''
    A function to calculate the energy generation rate.
    Accounts for contributions from both the pp-chain and the CNO cycle.
    '''
    # Define variables
    T7 = T/1e7
    T9 = T/1e9
    X1 = X # Mass fraction of H
    X_CNO = Z # Mass fraction of C+N+O (2/3 Z?)
    zeta = 1 # Order of unity from KWW
    Z1 = 1 # Nuclear charge of H
    Z2 = 1 # Nuclear charge of He
    # Estimate ψ from KWW figure 18.7
    def findPsi(T):
        T7 = T/1e7
        if T7 <= 1.5:
            psi = 1
            return psi
        elif T7 > 1.5 and T7 < 2.5:
            psi = 1.7
            return psi
        elif T7 >= 2.5:
            psi = 1.5
        return psi
    # Calculate weak screening
    E_DdivkT = 5.92e-3 * Z1 * Z2 * (zeta*rho/(T7**3))**0.5
    f11 = np.exp(E_DdivkT)
    # Calculate gaunt factor
    g11 = (1 + 3.82*T9 + 1.51*(T9**2) + 0.144*(T9**3) - 0.0114*(T9**4))
    g141 = (1-2.00*T9+3.41*(T9**2)-2.43*(T9**3))
    # Calculate energy generation rate for proton-proton chain
    if findPsi(T7) == None:
        ePP = np.nan
    else:
        ePP = (2.57e4 * findPsi(T7) * f11 * g11 * rho * (X1**2) * (T9**(-2/3)) * np.exp(-3.381/(T9**(1/3)))) #* u.erg/(u.g * u.s)
    # Calculate energy generation rate for CNO cycle
    eCNO = 8.24e25 * g141 * X_CNO * X1 * rho * np.power(T9, -2/3) * np.exp(-15.231*np.power(T9, -1/3)-np.square(T9/0.8))
    # Total energy generation rate
    eH = ePP + eCNO
    if verbose:
        print(f'Calculating energy generation rate...\n   ε_H = {eH:.2e} erg g^-1 s^-1')
    return ePP, eCNO, eH

# Set up density calculation
def findDensity(P, T, verbose=False):
    Prad = np.divide(a*np.power(T, 4), 3)
    Pgas = P-Prad #np.divide((rho*NA*k*T), mu)
    # Hard code for pressure inequality leading to negative (unphysical) density
    if Pgas < 0:
        Pgas = P
    rho = np.divide((Pgas)*mu, NA*k*T)
    if verbose:
        print('Calculating density...')
        print(f'Warning: Requires minimum input pressure of {Prad:.2e} dyne/cm^2')
        print(f'   Input pressure: {P:.2e} dyne/cm^2')
        print(f'   Input temperature: {T:.2e} K')
    if rho < 0:
        if verbose:
            print('   Total pressure less than radiation pressure!')
            print('   UNPHYSICAL DENSITY :(')
            print('Density calculation failed.')
        rho = np.nan
    if verbose:
        print(f'   Density: {rho:.2e} g/cm^3\n')
    return rho

# Set up nature of energy transport calculation
def findDel(M_r, vector, verbose=False):
    P, R, L, T = vector
    rho = findDensity(P, T)
    kappa = findOpacity(LogT=np.log10(T), Logrho=np.log10(rho))
    # Get actual del to figure out if region is radiative or convective (
    F_tot = np.divide(4*np.pi*np.square(R), L)
    del_rad = np.divide(3*np.square(R), 4*a*c*G)*np.divide(P*kappa, np.power(T, 4))*np.divide(F_tot, M_r) # KWW 4.30
    # Assuming fully ionised ideal gas
    Gamma2 = 5/3
    del_ad = np.divide(Gamma2-1, Gamma2) # HKT 3.94
    # From HKT 7.8-7.9
    if del_rad <= del_ad:
        del_ac = del_rad # Pure diffusive radiative transfer (or conduction)
        if verbose:
            print('\nEnergy transfer is RADIATIVE!')
        return del_rad
    else:
        del_ac = del_ad # Adiabatic convection
        if verbose:
            print('\nEnergy transfer is CONVECTIVE!')
        return del_ad
