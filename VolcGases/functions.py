import numpy as np
from scipy import optimize
import sys
import ctypes as ct
import numba as nb
import os
import platform

rootdir = os.path.dirname(os.path.realpath(__file__))+'/'
if platform.uname()[0] == "Windows":
    name = "libvolcgases.dll"
elif platform.uname()[0] == "Linux":
    name = "libvolcgases.so"
else:
    name = "libvolcgases.dylib"
libvolcgases = ct.CDLL(rootdir+name)
solve_gases_c = libvolcgases.solve_gases_c
solve_gases_c.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, \
                          ct.c_void_p, ct.c_void_p, ct.c_void_p, \
                          ct.c_void_p, ct.c_void_p, ct.c_void_p, \
                          ct.c_void_p, ct.c_void_p, ct.c_void_p]
solve_gases_c.restype = None

@nb.njit(nb.types.UniTuple(nb.float64,8)(nb.float64,nb.float64,nb.float64,nb.float64,nb.float64))
def solve_gases(T,P,f_O2,mCO2tot,mH2Otot):
    '''This function solves for the speciation of gases produced by
    a volcano. This code assumes magma composition of the lava erupting at
    Mt. Etna Italy.

    Inputs:
    T = temperature of the magma and gas in kelvin
    P = pressure of the gas in bar
    f_O2 = oxygen fugacity of the melt
    mCO2tot = mass fraction of CO2 in the magma
    mH2Otot = mass fraction of H2O in the magma

    Outputs:
    an array which contains
    [PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O]
    where
    PH2O = partial pressure of H2O in the gas in bar
    PH2 = partial pressure of H2 in the gas in bar
    PCO2 = partial pressure of CO2 in the gas in bar
    PCO = partial pressure of CO in the gas in bar
    PCH4 = partial pressure of CH4 in the gas in bar
    alphaG = moles of gas divide by total moles in gas and magma combined
    xCO2 = mol fraction of the CO2 in the magma
    xH2O = mol fraction of the H2O in the magma
    '''
    T_,P_,f_O2_,mCO2tot_,mH2Otot_ = \
    np.array(T,np.double),np.array(P,np.double), np.array(f_O2,np.double), np.array(mCO2tot,np.double), np.array(mH2Otot,np.double)
    
    P_H2O,P_H2,P_CO2,P_CO,P_CH4,alphaG,x_CO2,x_H2O = \
    np.array(0.0,np.double), np.array(0.0,np.double), np.array(0.0,np.double), np.array(0.0,np.double), \
    np.array(0.0,np.double), np.array(0.0,np.double), np.array(0.0,np.double), np.array(0.0,np.double)

    ierr = np.array(0,np.int32)
    
    solve_gases_c(T_.ctypes.data, P_.ctypes.data ,f_O2_.ctypes.data ,mCO2tot_.ctypes.data, mH2Otot_.ctypes.data, \
                  P_H2O.ctypes.data, P_H2.ctypes.data ,P_CO2.ctypes.data, \
                  P_CO.ctypes.data, P_CH4.ctypes.data, alphaG.ctypes.data, \
                  x_CO2.ctypes.data, x_H2O.ctypes.data, ierr.ctypes.data)
    if ierr.item() != 0:
        raise Exception('solve_gases failed because of unphysical inputs or non-linear solve failure.')
    
    return P_H2O.item(),P_H2.item(),P_CO2.item(),P_CO.item(),P_CH4.item(),alphaG.item(),x_CO2.item(),x_H2O.item()

# for backwards compatibility
solve_gases_jit = solve_gases

def solve_gases_nitrogen(T,P,f_o2,mCO2tot,mH2Otot,mN2tot):
    F1 = np.log(1.0/(M_CO2*x*1.0e6))+C_CO2*P/T+A1
    F2 = np.log(1.0/(M_H2O*x*100.0))+C_H2O*P/T+A2

    #calculate mol fraction of CO2 and H2O in the magma
    xCO2tot = (mCO2tot/M_CO2)/x
    xH2Otot = (mH2Otot/M_H2O)/x
    xN2tot = (mN2tot/M_N2)/x

    #equilibrium constants
    #made with Nasa thermodynamic database (Burcat database)
    K1 = np.exp(-29755.11319228574/T+6.652127716162998)
    K2 = np.exp(-33979.12369002451/T+10.418882755464773)
    K3 = np.exp(-96444.47151911151/T+0.22260815074146403)

    #constants
    C1 = K1/f_O2**0.5
    C2 = K2/f_O2**0.5
    C3 = K3/f_O2**2.0

    M_H2O = 18.01528 #Molar mass of H2O (g/mol)
    M_CO2 = 44.01 #Molar mass of CO2 (g/mol)
    M_N2 = 28.02 #Molar mass of N2 (g/mol)

    #Solubility constant for CO2 for Mt. Etna composition.
    #It is equal to the stuff after the second term in Eq. A1 in [1]
    A1 = -0.4200250000201988 
    #Solubility constant for H2O for Mt. Etna composition.
    #It is equal to the stuff after the second term in Eq. A2 in [1]
    A2 = -2.59560737813789
    #Inverse of molar mass of magma (mol of magma / g of magma).
    #See Table 1 in [1]
    x = 0.01550152865954013

    C_CO2 = 0.14 #!! Solubility constant. See Table 4 in [1] 
    C_H2O = 0.02 #!! Solubility constant. See Table 6 in [2]
    a_H2O = 0.54 #!! Solubility constant. See Table 1 in [1]
    a_CO2 = 1.0 #!! Solubility constant. See Table 1 in [1]
    d_H2O = 2.3 #!! Solubility constant. See Table 1 in [1]

    [A,B,C] = [27215,6.57,0.0552]
    f_o2_IW = 10**(-A/T + B + C*(P-1)/T)
    dIW = np.log10(fo2) - np.log10(f_o2_IW)

    XXSiO2 = 0.4795
    XXAl2O3 = 0.1732
    XXTiO2 = 0.0167

    args = P,F1,F2,xCO2tot,xH2Otot,xN2tot,C1,C2,C3,dIW,a_H2O,a_CO2,d_H2O,T

    P_H2O,P_H2,P_CO2,P_CO,P_CH4,alphaG,x_CO2,x_H2O = solve_gases(T,P,f_o2,mCO2tot,mH2Otot) 
    #guess using previous solve_gases (simpler system)
    #where do these values go?

    sol = optimize.root(fcn1_nitrogen, args = args, method = 'hybr') #solving for nitrogen in melt
    #finds zero values of equations in fcn1_nitrogen
    assert sol.success 
    
    ln_x_H2O, ln_x_CO2, ln_H2O, ln_CO2, lnalphaG, ln_H2, ln_CH4, ln_CO, ln_x_N2, ln_N2 = sol.x

    return np.exp(ln_x_H2O), np.exp(ln_x_CO2), np.exp(ln_H2O), np.exp(ln_CO2), np.exp(lnalphaG), np.exp(ln_H2), np.exp(ln_CH4), np.exp(ln_CO), np.exp(ln_x_N2), np.exp(ln_N2)
    #gases in melt in bar

def fcn1_nitrogen(y,args):
    #x is in the melt, mol fraction
    ln_x_H2O, ln_x_CO2, ln_H2O, ln_CO2, lnalphaG, ln_H2, ln_CH4, ln_CO, ln_x_N2, ln_N2 = y
    P,F1,F2,xCO2tot,xH2Otot,xN2tot,C1,C2,C3,dIW,a_H2O,a_CO2,d_H2O, T = args
    out = np.empty(10)
    #all of these equations == 0
    out[0] = np.exp(ln_H2O)+np.exp(ln_CO2)+np.exp(ln_H2)+np.exp(ln_CH4)+np.exp(ln_CO)+np.exp(ln_N2)-P #need to modify total pressure
    out[1] = -ln_x_CO2+np.exp(ln_x_H2O)*d_H2O+a_CO2*ln_CO2+F1
    out[2] = -ln_x_H2O+a_H2O*ln_H2O+F2
    out[3] = -xH2Otot*P + (np.exp(ln_H2O)+np.exp(ln_H2)+2*np.exp(ln_CH4))*np.exp(lnalphaG)+(1-np.exp(lnalphaG))*np.exp(ln_x_H2O)*P
    out[4] = -xCO2tot*P + (np.exp(ln_CO2)+np.exp(ln_CO)+np.exp(ln_CH4))*np.exp(lnalphaG)+(1-np.exp(lnalphaG))*np.exp(ln_x_CO2)*P
    out[5] = np.log(C1)+ln_H2O-ln_H2
    out[6] = np.log(C2)+ln_CO2-ln_CO
    out[7] = np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4
    #adding nitrogen, pressure in GPa
    pN2_Gpa = pn2 / 1e-4
    pt_Gpa = P / 1e-4
    M_magma = 1/0.01550152865954013
    N_ppm = (np.exp(ln_x_N2) * M_N2 / M_magma)*1e6 #to ppm (eqn 3 Nicks paper)
    #solubility of N2 in melt
    out[8] = -N_ppm + np.sqrt(pN2_Gpa) * np.exp(5908*pt_Gpa**0.5/T - 1.6*dIW) + pN2_Gpa*np.exp(4.67 + 7.11 * XXSiO2 - 13.06*XXAl2O3 - 120.67*XXTiO2)
    #conservation equation for N2 check this
    out[9] = -xN2tot*P + np.exp(ln_N2)*np.exp(lnalphaG) + (1-np.exp(lnalphaG))*np.exp(ln_x_N2)*P #(after eqn 11,12)
    return out 


def degassing_pressure(T,DFMQ,mCO2tot,mH2Otot,P_range = [1e-4,30000]):
    """
    This function determines the overburden pressure where degassing begins.

    Inputs:
    T = temperature of the magma and gas in kelvin
    DFMQ = oxygen fugacity of the melt relative to the FMQ buffer (e.g. FMQ+1 -> 1)
    mCO2tot = mass fraction of CO2 in the magma
    mH2Otot = mass fraction of H2O in the magma
    P_range = the degassing pressure must fall within this range

    Outputs:
    P = Pressure where degassing begins
    """
    A = 25738
    B = 9
    C = 0.092
    def func(P):
        log_FMQ = (-A/T+B+C*(P-1)/T)
        f_O2 = 10**(log_FMQ+DFMQ) # units - bar
        P_H2O,P_H2,P_CO2,P_CO,P_CH4,alphaG,x_CO2,x_H2O = solve_gases(T,P,f_O2,mCO2tot,mH2Otot)
        return alphaG
    P = find_first_zero(func, P_range[0], P_range[1])
    return P

def find_first_zero(func, min, max, tol=1e-5):
    min, max = float(min), float(max)
    assert (max + tol) > max
    while (max - min) > tol:
        mid = (min + max) / 2
        sol = func(mid)
        if sol == 0 or sol != sol:
            max = mid
        else:
            min = mid
    return max

@nb.njit()
def gas_production(T,P,f_O2,mCO2tot,mH2Otot):
    """
    This function calculates gas production (mol gas produce/kg of magma)
    of degassing volcano.

    Inputs:
    T = temperature of the magma and gas in kelvin
    P = pressure of the gas in bar
    f_O2 = oxygen fugacity of the melt (bar)
    mCO2tot = mass fraction of CO2 in the magma
    mH2Otot = mass fraction of H2O in the magma

    Outputs:
    an array which contains
    [H2O, H2, CO2, CO, CH4]
    where
    H2O = H2O production (mol H2O produce/kg of magma)
    H2 = H2 production (mol H2 produce/kg of magma)
    CO2 = CO2 production (mol CO2 produce/kg of magma)
    CO = CO production (mol CO produce/kg of magma)
    CH4 = CH4 production (mol CH4 produce/kg of magma)
    """
    x = 0.01550152865954013 # mol of magma/ g of magma
    P_H2O,P_H2,P_CO2,P_CO,P_CH4,alphaG,x_CO2,x_H2O = solve_gases(T,P,f_O2,mCO2tot,mH2Otot)
    H2O = 1000*alphaG*x*P_H2O/P
    H2 = 1000*alphaG*x*P_H2/P
    CO2 = 1000*alphaG*x*P_CO2/P
    CO = 1000*alphaG*x*P_CO/P
    CH4 = 1000*alphaG*x*P_CH4/P
    return np.array([H2O,H2,CO2,CO,CH4])

@nb.njit()
def outgassing_flux(T,P,f_O2,mCO2tot,mH2Otot,Q):
    """
    This function calculates gas production (mol gas produce/kg of magma)
    of degassing volcano.

    Inputs:
    T = temperature of the magma and gas in kelvin
    P = pressure of the gas in bar
    f_O2 = oxygen fugacity of the melt (bar)
    mCO2tot = mass fraction of CO2 in the magma
    mH2Otot = mass fraction of H2O in the magma
    Q = magma production rate (kg magma/yr)

    Outputs:
    an array which contains
    [FH2O, FH2, FCO2, FCO, FCH4]
    where
    FH2O = volcanic H2O flux (mol H2O/yr)
    FH2 = volcanic H2 flux (mol H2/yr)
    FCO2 = Cvolcanic O2 flux (mol CO2/yr)
    FCO = volcanic CO flux (mol CO/yr)
    FCH4 = volcanic CH4 flux (mol CH4/yr)
    """

    H2O,H2,CO2,CO,CH4 = gas_production(T,P,f_O2,mCO2tot,mH2Otot)
    FH2O = H2O*Q
    FH2 = H2*Q
    FCO2 = CO2*Q
    FCO = CO*Q
    FCH4 = CH4*Q
    return np.array([FH2O,FH2,FCO2,FCO,FCH4])

def closed_system_cooling(T,P,f_O2,mCO2tot,mH2Otot,DT):
    """
    This function calculates the chemistry of volcanic gas after
    closed system cooling. It first calculates gas chemistry when in equilibrium
    with the magma. Then it calculates the equlibrium chemistry of that gas if it cooled by
    temperature DT.

    Inputs:
    T = temperature of the magma and gas in kelvin
    P = pressure of the gas in bar
    f_O2 = oxygen fugacity of the melt (bar)
    mCO2tot = mass fraction of CO2 in the magma
    mH2Otot = mass fraction of H2O in the magma
    DT = temperature drop of gas in kelvin. positive number = drop

    Outputs:
    an array which contains
    [N_H2O, N_H2, N_CO2, N_CO, N_CH4, N_O2]
    where
    N_H2O = H2O production (mol H2O produce/kg of magma)
    N_H2 = H2 production (mol H2 produce/kg of magma)
    N_CO2 = CO2 production (mol CO2 produce/kg of magma)
    N_CO = CO production (mol CO produce/kg of magma)
    N_CH4 = CH4 production (mol CH4 produce/kg of magma)
    N_O2 = O2 production (mol O2 produce/kg of magma)
    """

    T1 = T-DT
    x = 0.01550152865954013

    # calculate gas composition when in equilibrium with the magma
    P_H2O,P_H2,P_CO2,P_CO,P_CH4,alphaG,x_CO2,x_H2O = solve_gases(T,P,f_O2,mCO2tot,mH2Otot)
    if alphaG==0:
        return np.array([0,0,0,0,0,0])
    P_i = np.array([P_H2O,P_H2,P_CO2,P_CO,P_CH4,f_O2])
    LH2O,LH2,LCO2,LCO,LCH4,LO2 = 0,1,2,3,4,5

    # now calculate moles of each gas per kg of magma (N is moles/kg of magma!)
    N_i = 1000*alphaG*x*P_i/P

    # calculate total moles of each atom
    NH_tot = 2*N_i[LH2]+2*N_i[LH2O]+4*N_i[LCH4]
    NC_tot = N_i[LCO2]+N_i[LCO]+N_i[LCH4]
    NO_tot = N_i[LH2O]+2*N_i[LCO2]+N_i[LCO]+2*N_i[LO2]

    # everying in terms of logs
    def closed_system(y):
        lnN_H2O,lnN_H2,lnN_CO2,lnN_CO,lnN_CH4,lnN_O2,lnN_tot = y
        return (lnN_H2+0.5*lnN_O2+.5*np.log(P)-(np.log(K1)+lnN_H2O+0.5*lnN_tot),\
                lnN_CO+0.5*lnN_O2+.5*np.log(P)-(np.log(K2)+lnN_CO2+0.5*lnN_tot),\
                lnN_CH4+2*lnN_O2-(np.log(K3)+lnN_CO2+2*lnN_H2O),\
                NH_tot-(2*np.exp(lnN_H2O)+2*np.exp(lnN_H2)+4*np.exp(lnN_CH4)),\
                NO_tot-(np.exp(lnN_H2O)+np.exp(lnN_CO)+2*np.exp(lnN_CO2)+2*np.exp(lnN_O2)),\
                NC_tot-(np.exp(lnN_CO2)+np.exp(lnN_CO)+np.exp(lnN_CH4)),\
                np.exp(lnN_tot)-(np.exp(lnN_H2O)+np.exp(lnN_H2)+np.exp(lnN_CO2)+np.exp(lnN_CO)+np.exp(lnN_CH4)+np.exp(lnN_O2)))


    # intital condition is equilibrium state of gases in magma
    init_cond = np.log(np.array([N_i[LH2O],N_i[LH2],N_i[LCO2],N_i[LCO],N_i[LCH4],N_i[LO2],np.sum(N_i)]))

    # equilibrium constants at new temperature T1
    K1 = np.exp(-29755.11319228574/T1+6.652127716162998)
    K2 = np.exp(-33979.12369002451/T1+10.418882755464773)
    K3 = np.exp(-96444.47151911151/T1+0.22260815074146403)

    sol = optimize.root(closed_system,init_cond,method='lm',options={'maxiter': 500000})
    # check for error
    error = np.linalg.norm(closed_system(sol['x']))
    tol = 1e-7
    if sol['success']== False or error>tol:
        print(error)
        print(sol)
        sys.exit('root finding for closed system failed')

    # solution. Should be moles of gas/kg of magma
    N_sol = np.exp(sol['x'])[:-1]
    N_tot = np.exp(sol['x'])[-1]
    N_H2O, N_H2, N_CO2, N_CO, N_CH4, N_O2 = N_sol

    return np.array([N_H2O, N_H2, N_CO2, N_CO, N_CH4, N_O2])
