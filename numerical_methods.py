from scipy.integrate import solve_ivp
from scipy.optimize import newton, least_squares, root, minimize

from physical_quantities import *

# Set up calculation of boundary values at stellar centre
def findCoreBVs(P_c, T_c, M_frac=1e-12, verbose=False):
    '''
    A function to calculate boundary values for the four dependent
    stellar variables at mass point m very slightly away from the
    stellar centre. For use when integrating outwards.
    '''
    if verbose:
        print('Calculating initial values for P, r, l, T at stellar core...\n')
    # Define enclosed mass at point slightly away from stellar centre
    m = M_star*M_frac
    # Find density
    rho_c = findDensity(P_c, T_c, verbose=verbose)
    if np.isnan(rho_c):
        if verbose:
            print('\nFailed to calculate BVs at core---unphysical density.')
        return np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        Logrho_c = np.log10(rho_c)
        # Find radius, from KWW 11.3
        r = np.power(np.divide(3, 4*np.pi*rho_c), 1/3) * np.power(m, 1/3)
        # Find pressure, from KWW 11.6
        P = (P_c - np.divide(3*G, 8*np.pi)*np.power((np.divide(4*np.pi, 3)*rho_c), 4/3)*np.power(m, 2/3))
        # Use energy generation rate to determine lumnosity and temperature
        egr = findEGR(T_c, rho_c, verbose=verbose)
        if egr[0] > egr[1]:
            if verbose:
                print('\nMethod of energy transport?')
                print(f'   PP: {egr[0]:.2e} > CNO: {egr[1]:.2e} ---- Core is RADIATIVE!\n')
            l = egr[2]*m # KWW 11.4
            kappa_c = 10**findOpacity(LogT=np.log10(T_c), Logrho=np.log10(rho_c), verbose=verbose)
            T = np.power((np.power(T_c, 4)-(1/2*a*c)*np.power((3/4*np.pi), 2/3)*kappa_c*egr[0]*np.power(rho_c, 4/3)*np.power(m, 2/3)), 1/4) # KWW 11.9
        else:
            if verbose:
                print('\nMethod of energy transport?')
                print(f'   PP: {egr[0]:.2e} < CNO: {egr[1]:.2e} ---- Core is CONVECTIVE!')
            l = egr[2]*m # KWW 11.4
            T = np.exp(np.log(T_c)-(np.power((np.pi/6), 1/3)*G*np.divide(del_ad*np.power(rho_c, 4/3), P_c)*np.power(m, 2/3))) # KWW 11.9
        # Get vector of initial values
        vector = np.asarray([P, r, l, T])
        if verbose:
            print(f'\nINITIAL VALUES (CORE):\n   Pressure = {P:.2e} g/cm*s^2\n   Radius = {r:.2e} cm\n   Luminosity = {l:.2e} erg/s\n   Temperature = {T_c:.2e} K\n')
        return vector

# Set up calculation of surface boundary values
def findSurfaceBVs(M_star, R_star, L_star, M_frac=0.9999, verbose=False):
    '''
    Calculates boundary values for P, R, L, T roughly at the stellar surface.
    For use when integrating inwards.
    '''
    # Set enclosed mass
    m = M_star*M_frac
    # Find effective temperature, from KWW 11.14
    T_eff = (np.divide(L_star, 4*np.pi*np.square(R_star)*sigma_sb))**0.25
    # Find surface gravity
    g_s = G*M_star/(R_star**2)
    # Find minimum density where opacity pressure and gas+radiation
    # are equivalent
    def findDensityMin(rho):
        kappa = findOpacity(np.log10(T_eff), np.log10(rho))
        # From HKT 4.48
        P_tau = (2/3)*(g_s/kappa) * (1+(kappa*L_star/(4*np.pi*c*G*M_star)))
        # From HKT 3.17 and 3.28
        P_gasrad = (1/3)*a*(T_eff**4) + rho*NA*k*T_eff/mu
        diff = 1 - P_tau/P_gasrad
        return np.abs(diff**2)
    rho_sol = minimize(findDensityMin, 1e-8, args=(), method='Nelder-Mead', bounds=[(1e-13,1e-5)])
    if rho_sol.success:
        rho = rho_sol.x[0]
    else:
        if verbose:
            print('Unsuccessful density calculation for this surface temperature.')
        rho = np.nan
    # Find surface opacity
    kappa = findOpacity(LogT=np.log10(T_eff), Logrho=np.log10(rho))
    # Find surface pressure
    P_s = 2*g_s/(3*kappa) * (1 + (kappa*L_star/(4*np.pi*c*G*M_star)))
    # Get vector of initial values
    vector = np.asarray([P_s, R_star, L_star, T_eff])
    if verbose:
        print(f'\nINITIAL VALUES (SURFACE):\n    Pressure = {P_s:.2e} g cm^-1 s^-2\n    Radius = {R_star:.2e} cm\n    Luminosity = {L_star:.2e} erg s^-1\n    Temperature = {T_eff:.2e} K\n')
    return vector

# Set up function to calculate derivatives for the four ODEs
def findDerivs(M_r, vector, verbose=False):
    '''
    Takes independent variable M_r and four dependent variables
    P, r, l, and T and returns the derivatives given by the four
    coupled ODEs of stellar structure:

    1. dP/dm = -Gm/4πr^4
    2. dr/dm = 1/4πr^2ρ
    3. dl/dm = ε
    4. dT/dm = -(GmT/4πr^4ρ)*∇
    '''
    # Define the four dependent variables
    P, r, l, T = vector
    # Get density
    rho = findDensity(P, T, verbose=verbose)
    if np.isnan(rho):
        return np.array([np.nan, np.nan, np.nan, np.nan])
    kappa = 10**findOpacity(LogT=np.log10(T), Logrho=np.log10(rho), verbose=verbose)
    # Get actual del to figure out if region is radiative or convective (
    del_ac = findDel(M_r, vector, verbose=verbose)
    # Get four coupled ODEs
    dPdm = np.divide(-G*M_r, 4*np.pi*np.power(r, 4))
    drdm = 1/(4*np.pi*np.square(r)*rho)
    dldm = findEGR(T, rho)[2]
    dTdm = np.divide(-G*M_r*T, 4*np.pi*np.power(r, 4)*rho)*del_ac
    return np.asarray([dPdm, drdm, dldm, dTdm])

# Set up function to solve the four coupled ODEs of stellar structure and evolution
def solveODEs(guesses, M_star, M_fracs, verbose=False, steps=1e5):
        '''
        Similar to 'shootf' function from KWW. Uses scipy.integrate.solve_ivp()
        to solve initial value problem of the four coupled stellar structure ODEs.
        Steps:
        (1) Runs two integrations:
                (a) Core outwards to fitting point
                (b) Surface inwards to fitting point
        (2) Calculates mismatch between the two solutions at the fitting point.
        '''
        # Define initial guesses for stellar variable values
        P_c, R_star, L_star, T_c = guesses
        # Define enclosed mass fractions
        M_frac1, M_frac2, M_fracf = M_fracs
        # Define enclosed mass near core and near surface
        m_core = M_star*M_frac1
        m_surf = M_star*M_frac2
        m_fit = M_star*M_fracf
        # Define steps at which to evaluate integration
        m_outwards = np.linspace(m_core, m_fit, num=int(steps/2))
        m_inwards = np.linspace(m_surf, m_fit, num=int(steps/2))
        # Find initial values near core and near surface
        BVs_core = findCoreBVs(P_c=P_c, T_c=T_c, M_frac=M_frac1, verbose=verbose)
        BVs_surf = findSurfaceBVs(M_star, R_star, L_star, M_frac=M_frac2, verbose=verbose)
        if np.isnan(sum(BVs_core)):
            return np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            # Solve integration from core outwards to fitting point
            print('integrating outwards...')
            solve_c2s = solve_ivp(fun=findDerivs,
                                  t_span=(m_core, m_fit),
                                  y0=BVs_core,
                                  method='RK45', t_eval=m_outwards)
            c2s_sols = solve_c2s.y
        if np.isnan(sum(BVs_surf)):
            return np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            # Solve integration from surface inwards to fitting point
            print('integrating inwards...')
            solve_s2c = solve_ivp(fun=findDerivs,
                                  t_span=(m_surf, m_fit),
                                  y0=BVs_surf,
                                  method='RK45', t_eval=m_inwards)
            s2c_sols = solve_s2c.y
        # Compute residuals for solutions at the fitting point
        dP = np.divide(c2s_sols[0, -1]-s2c_sols[0, -1], P_c)
        dr = np.divide(c2s_sols[1, -1]-s2c_sols[1, -1], R_star)
        dl = np.divide(c2s_sols[2, -1]-s2c_sols[2, -1], L_star)
        dT = np.divide(c2s_sols[3, -1]-s2c_sols[3, -1], T_c)
        residuals = np.asarray([dP, dr, dl, dT])
        if verbose:
            print(f'Residuals:\n    dP = {dP}\n    dr = {dr}\n    dl = {dl}\n    dT = {dT}')
        return residuals

# Set up function to converge stellar parameters at fitting point
def findConvergence(guesses, M_star, M_fracs, verbose=False):
        '''
        Similar to 'newt' function from KWW. Repeatedly calls the
        ODE solver function (solveODEs) to calculate updated boundary values and
        obtain a converged solution.
        '''
        P, R, L, T = guesses
        args = (M_star, M_fracs, verbose)
        lower_bounds = [P*1e-1, R*1e-1, L*1e-1, T]
        upper_bounds = [P*1e2, R*1e1, L*1e1, T*1e3]
        bounds = (lower_bounds, upper_bounds)
        converged_sol = least_squares(fun=solveODEs,
                                  x0=guesses,
                                  args=args,
                                  ftol=1e-6,
                                  xtol=None,
                                  gtol=None,
                                  bounds=bounds)
        return converged_sol

def findSolution(vector, M_star, M_fracs, steps=int(1e5), verbose=False):
    '''
    A function that inputs the vector of stellar parameters from the converged
    model and uses them to solve the four coupled ODEs of stellar structure
    and evolution.
    '''
    P_c, R_star, L_star, T_c = vector # converged_sol.x
    M_frac1, M_frac2, M_fracf = M_fracs
    # Find boundary values at stellar core and surface
    BVs_core = findCoreBVs(P_c, T_c, M_frac=M_frac1)
    BVs_surf = findSurfaceBVs(M_star, R_star, L_star, M_frac=M_frac2)
    # Define enclosed masses
    m_core = M_star*M_frac1
    m_surf = M_star*M_frac2
    m_fit = M_star*M_fracf
    # Define steps at which to evaluate integration
    m_outwards = np.linspace(m_core, m_fit, num=int(steps/2))
    m_inwards = np.linspace(m_surf, m_fit, num=int(steps/2))
    # Protect against bad BVs
    if np.isnan(sum(BVs_core)):
        return np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        # Solve integration from core outwards to fitting point
        print('integrating outwards...')
        solve_c2s = solve_ivp(fun=findDerivs,
                                  t_span=(m_core, m_fit),
                                  y0=IVs_core,
                                  method='RK45', t_eval=m_outwards)
        c2s_sols = solve_c2s.y
    # Protect against bad BVs
    if np.isnan(sum(IVs_surf)):
        return np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        # Solve integration from surface inwards to fitting point
        print('integrating inwards...')
        solve_s2c = solve_ivp(fun=findDerivs,
                                  t_span=(m_surf, m_fit),
                                  y0=IVs_surf,
                                  method='RK45', t_eval=m_inwards)
        s2c_sols = solve_s2c.y
    # Concatenate results into a table
    mass = np.concatenate([m_outwards, np.flipud(m_inwards)], axis=0)
    final_sol = np.zeros((6, mass.shape[0]))
    final_sol[0] = mass
    param_sols = np.concatenate([m_outwards, np.fliplr(m_inwards)], axis=1)
    final_sol[1:5] = param_sols
    # Add density to the table
    rho = []
    for i in np.arange(len(final_sol[1])):
        ind_rho = findDensity(final_sol[1][i], final_sol[4][i])
        rho.append(ind_rho)
    final_sol[5] = rho
    # Add nature of energy transport to the table
    nabla = findDel(mass, param_sols)
    final_sol[6] = nabla
    return solution
