# N. M. Rathmann <rathmann@nbi.ku.dk>, 2023

import numpy as np
from progress.bar import Bar

def lagrangianparcel(sf, mod, strain_target, Nt=100, dt=None, nlm0=None, verbose=True, \
                    iota=None, zeta=0, Lambda=None, Gamma0=None, # fabric process params \
                    nu=None, regexpo=None, apply_bounds=False # regularization \
    ):

    """
    Lagrangian parcel integrator subject to a time-constant mode of deformation (constant ugrad)
    """

    ### Mode of deformation
    
    if not ('T' in mod): mod['T'] = 1 # set T if not characteristic time-scale provided
    
    if mod['type'] == 'simpleshear' or mod['type'] == 'ss':
        # strain_target is shear angle (rad)
        if dt is None: dt = sf.simpleshear_gamma_to_t(strain_target, mod['T'])/Nt
        F     = np.array([sf.simpleshear_F(mod['plane'], mod['T'], dt*nt) for nt in np.arange(Nt+1)])
        ugrad = sf.simpleshear_ugrad(mod['plane'], mod['T'])
        
    elif mod['type'] == 'pureshear' or mod['type'] == 'ps':
        # strain_target is axial strain
        if dt is None: dt = sf.pureshear_strainii_to_t(strain_target, mod['T'])/Nt
        F     = np.array([sf.pureshear_F(mod['axis'], mod['r'], mod['T'], dt*nt) for nt in np.arange(Nt+1)])
        ugrad = sf.pureshear_ugrad(mod['axis'], mod['r'], mod['T'])
        
    elif mod['type'] == 'rigidrotation' or mod['type'] == 'rr':
        # strain_target is rotation angle
        raise Exception('type="rigidrotation" not yet supported')
        
    else:
        raise ValueError('Mode of deformation type="%s" not supported'%(mod['type']))
    
    D, W = sf.ugrad_to_D_and_W(ugrad) 
    S = D.copy() # assume coaxial stress strain-rate
    time = dt*np.arange(Nt+1)    

    ### State vector

    nlm_len = sf.nlm_len()
    nlm  = np.zeros((Nt+1,nlm_len), dtype=np.complex64)
    if nlm0 is None: nlm[0,0] = 1/np.sqrt(4*np.pi) # initially isotropic distribution
    else:            nlm[0,:] = nlm0 # initial state provided by caller
    nlm_dummy = nlm[0,:].copy()
    
    ### Steady CPO operators

    M_zero = np.zeros((nlm_len,nlm_len), dtype=np.complex64)

    # Regularization
    if regexpo is None: M_REG = nu*sf.M_REG(nlm_dummy, D) if nu is not None else M_zero
    else:               M_REG = M_REG_custom(nu, regexpo, D, sf)
    
    # CDRX
    M_CDRX = Lambda*sf.M_CDRX(nlm_dummy) if Lambda is not None else M_zero

    # Lattice rotation
    M_LROT = sf.M_LROT(nlm_dummy, D, W, iota, zeta) if iota is not None else M_zero
           
    ### Euler integration

    if verbose: bar = Bar('MOD=%s :: Nt=%i :: dt=%.2e :: nlm_len=%i ::'%(mod['type'],Nt,dt,nlm_len), max=Nt-1, fill='#', suffix='%(percent).1f%% - %(eta)ds')
    
    for nt in np.arange(1,Nt+1):
        nlm_prev = nlm[nt-1,:]
        M = M_LROT + M_CDRX + M_REG
        M += Gamma0*sf.M_DDRX(nlm_prev, S) if Gamma0 is not None else M_zero
        nlm[nt,:] = nlm_prev + dt*np.matmul(M, nlm_prev)
        
        if verbose: bar.next()
            
    return nlm, F, time, ugrad
                
    
def M_REG_custom(nu, expo, D, sf):

    """
    Custom regularization operator
    """

    L = sf.Lcap
    nlm_len = int((L+1)*(L+2)/2)
    nlm_dummy = np.zeros((nlm_len), dtype=np.complex64)
    L = sf.Lmat(nlm_dummy)/(L*(L+1)) # normalized laplacian matrix
    ratemag = nu*np.linalg.norm(D)
    M_REG = -ratemag*np.power(np.abs(L), expo)
    
    return M_REG

