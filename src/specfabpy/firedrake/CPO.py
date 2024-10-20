r"""
Firedrake interface for CPO dynamics using specfab
"""

import numpy as np
from ..specfabpy import specfabpy as sf__
from .. import common as sfcom
from firedrake import *

class CPO:
    def __init__(
        self, mesh, boundaries, L, nu_multiplier=1, nu_realspace=1e-3, modelplane='xz', ds=None, nvec=None
    ):
        ### Setup
        self.mesh, self.boundaries = mesh, boundaries
        self.ds = ds   if ds   is not None else Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        self.n  = nvec if nvec is not None else FacetNormal(self.mesh)
        self.L = int(L) # spectral truncation
        self.USE_REDUCED = True # use reduced representation of fabric state vector

        self.modelplane = modelplane
        if self.modelplane not in ['xy', 'xz']:
            raise ValueError('modelplane "%s" should be either "xy" or "xz"'%(self.modelplane))

        # Real-space stabilization (multiplicative constant of real-space Laplacian)
        self.nu_realspace  = Constant(nu_realspace)
        # Multiplier of orientation-space regularization magnitude
        self.nu_multiplier = Constant(nu_multiplier)

        ### Initialize specfab
        self.sf = sf__
        self.lm, self.nlm_len_full = self.sf.init(self.L)
        # use reduced or full form?
        self.nlm_len = self.sf.get_rnlm_len() if self.USE_REDUCED else self.nlm_len_full

        eletype, eleorder = 'CG', 1 # ordinary linear elements
        self.Rele = FiniteElement(eletype, self.mesh.ufl_cell(), eleorder)
        # note: a VectorElement is nothing but a MixedElement combining (multiplying)
        # "dim" copies of a FiniteElement
        self.Vele = MixedElement([self.Rele]*self.nlm_len)
        self.R = FunctionSpace(self.mesh, self.Rele)
        self.V = VectorFunctionSpace(self.mesh, "CG", 1, dim=self.nlm_len)
        # strain-rate and spin function space
        self.G = TensorFunctionSpace(self.mesh, eletype, eleorder, shape=(2,2))
        # for MPI support, get vector() size like this. Else could have used:
        # numdofs = self.R.dim()
        self.numdofs = Function(self.R).vector().local_size()

        if self.modelplane=='xy':
            raise NotImplementedError("Haven't got to this yet")
            self.Wele = MixedElement([self.Vele, self.Vele]) # real, imag components
            self.W = FunctionSpace(self.mesh, self.Wele)
            self.dofs_re = np.array([self.W.sub(0).sub(ii).dofmap().dofs() for ii in range(self.nlm_len)])
            self.dofs_im = np.array([self.W.sub(1).sub(ii).dofmap().dofs() for ii in range(self.nlm_len)])
            # Test and trial functions
            self.pr, self.pi = TrialFunctions(self.W) # unknowns (real, imag part of nlm coefs)
            self.qr, self.qi = TestFunctions(self.W)  # weight functions (real, imag part of nlm coefs)
            # for easy access of each subelement of the mixed element (real, imag parts)
            self.qr_sub, self.qi_sub = split(self.qr), split(self.qi)
        elif self.modelplane=='xz':
            self.Wele = self.Vele
            self.W = self.V
            # TODO: Figure out how to do this correctly in Firedrake
            #self.dofs_re = np.array([self.W.sub(ii).dofmap().dofs() for ii in range(self.nlm_len)])

            # Test and trial functions
            self.pr = TrialFunction(self.W) # unknowns (real part of nlm coefs)
            self.qr = TestFunction(self.W)  # weight functions (real part of nlm coefs)
            self.pr_sub = split(self.pr)    # for easy access of each subelement
            self.qr_sub = split(self.qr)    # for easy access of each subelement

        self.w      = Function(self.W) # Current solution
        self.w_prev = Function(self.W) # Previous solution

        self.nlm_dummy = np.zeros((self.nlm_len_full))
        self.Mk_LROT     = [ [Function(self.V) for ii in np.arange(self.nlm_len)] for ii in range(4) ] # rr, ri, ir, ii
        self.Mk_DDRX_src = [ [Function(self.V) for ii in np.arange(self.nlm_len)] for ii in range(4) ]
        self.Mk_CDRX     = [ [Function(self.V) for ii in np.arange(self.nlm_len)] for ii in range(4) ]
        self.Mk_REG      = [ [Function(self.V) for ii in np.arange(self.nlm_len)] for ii in range(4) ]

        # Idealized states
        self.nlm_iso   = [1/np.sqrt(4*np.pi)] + [0]*(self.nlm_len-1) # Isotropic and normalized state
        self.nlm_zero  = [0]*(self.nlm_len)

    def initialize(self, wr=None, wi=None):
        """
        Initialize uniform CPO field
        """
        if wr is None: wr, wi = self.nlm_iso, self.nlm_zero
        if self.modelplane=='xy':
            self.w.sub(0).assign(project(Constant(wr), self.V)) # real part
            self.w.sub(1).assign(project(Constant(wi), self.V)) # imag part
        elif self.modelplane=='xz':
            self.w.assign(project(Constant(wr), self.V)) # real part

    def set_state(self, w, interp=False):
        if interp:
            raise ValueError(
                """CPO.set_state() supports only setting function space vars, not interpolating expressions or constants."""
            )
        else:
            self.w.assign(w)
            self.w_prev.assign(w)

    def set_BCs(self, wr, wi, domid):
        """
        Set boundary conditions
        """
        self.bcs = []

        for ii, did in enumerate(domid):
            if self.modelplane=='xy':
                self.bcs += [DirichletBC(self.W.sub(0), wr[ii], did)] # real part
                self.bcs += [DirichletBC(self.W.sub(1), wi[ii], did)] # imag part

            elif self.modelplane=='xz':
                # TODO: Check this
                self.bcs += [DirichletBC(self.W, wr[ii], did)] # real part

    def set_isotropic_BCs(self, domid):
        """
        Easy setting of isotropic boundary conditions
        """
        wr = [Constant(self.nlm_iso),]
        wi = [Constant(self.nlm_zero),]
        self.set_BCs(wr, wi, [domid,])

    def weakform(self, u, S, dt, iota, Gamma0, Lambda0, zeta=0, steadystate=False):
        """
        Build weak form from dynamical matrices
        """

        ENABLE_LROT = iota is not None
        ENABLE_DDRX = (Gamma0 is not None) and (S is not None)
        ENABLE_CDRX = Lambda0 is not None

        # Flattened strain-rate and spin tensors for accessing them per node
        Df = project( sym(grad(u)), self.G).vector()[:] # strain rate
        Wf = project(skew(grad(u)), self.G).vector()[:] # spin
        Sf = project(S, self.G).vector()[:] # deviatoric stress

        # Dynamical matrices at each DOF
        # NOTE: The array shapes that you get from Firedrake are different
        # from FEniCS so the indexing is different here.
        if ENABLE_LROT:
            M_LROT_nodal = np.array([self.sf.reduce_M(self.sf.M_LROT(self.nlm_dummy, self.mat3d(Df[nn]), self.mat3d(Wf[nn]), iota, zeta), self.nlm_len) for nn in np.arange(self.numdofs)] )
        #if ENABLE_DDRX: M_DDRX_src_nodal = np.array([self.sf.reduce_M(self.sf.M_DDRX_src(self.nlm_dummy, self.mat3d(Sf[nn*4:(nn+1)*4])), self.nlm_len) for nn in np.arange(self.numdofs)] )
        M_REG_nodal = np.array([self.sf.reduce_M(self.sf.M_REG(self.nlm_dummy, self.mat3d(Df[nn])), self.nlm_len) for nn in np.arange(self.numdofs)] )

        # Populate entries of dynamical matrices
        if   self.modelplane=='xy': krng = range(4) # rr, ri, ir, ii
        elif self.modelplane=='xz': krng = range(1) # rr
        for ii in np.arange(self.nlm_len):
            for kk in krng:
                if ENABLE_LROT: self.Mk_LROT[kk][ii].vector()[:] = M_LROT_nodal[:,kk,ii,:]
                if ENABLE_DDRX: self.Mk_DDRX_src[kk][ii].vector()[:] = M_DDRX_src_nodal[:,kk,ii,:]
                self.Mk_REG[kk][ii].vector()[:] = M_REG_nodal[:,kk,ii,:]

        ### Construct weak form
        dtinv = Constant(1/dt)

        if self.modelplane == 'xy':
            raise NotImplementedError("Haven't got here yet")
        elif self.modelplane=='xz':

            # Real space advection, div(s*u)
            F = dot(dot(u, nabla_grad(self.pr)), self.qr)*dx # real part

            # Time derivative
            if not steadystate:
                F += dtinv * dot( (self.pr-self.w_prev), self.qr)*dx # real part

            # Real space stabilization (Laplacian diffusion)
            F += self.nu_realspace * inner(grad(self.pr), grad(self.qr))*dx # real part

            # Lattice rotation
            if ENABLE_LROT:
                Mrr_LROT, *_ = self.Mk_LROT # unpack for readability
                F += -sum([ dot(Mrr_LROT[ii], self.pr)*self.qr_sub[ii]*dx for ii in np.arange(self.nlm_len)]) # real part

            # DDRX
            if ENABLE_DDRX:
                Mrr_DDRX_src, *_ = self.Mk_DDRX_src # unpack for readability
                F_src  = sum([ -Gamma0*dot(Mrr_DDRX_src[ii], self.pr)*self.qr_sub[ii]*dx for ii in np.arange(self.nlm_len)])
                # nonlinear sink term is linearized around previous solution (self.w_prev) following Rathmann and Lilien (2021)
                F_sink = sum([ -Gamma0*dot(Mrr_DDRX_src[0],self.w_prev)/self.w_prev[0]*self.pr_sub[ii] * self.qr_sub[ii]*dx for ii in np.arange(self.nlm_len)])
#                F_sink = sum([ -Gamma0*dot(Mrr_DDRX_src[0],self.pr)*self.pr_sub[ii] * self.qr_sub[ii]*dx for ii in np.arange(self.nlm_len)]) # nonlinear sink term is linearized around previous solution (self.w_prev) following Rathmann and Lilien (2021)
                F += F_src - F_sink

            # Orientation space stabilization (hyper diffusion)
            Mrr_REG,  *_ = self.Mk_REG  # unpack for readability
            F += -self.nu_multiplier * sum([ dot(Mrr_REG[ii], self.pr)*self.qr_sub[ii]*dx for ii in np.arange(self.nlm_len)]) # real part

        return F

    def mat3d(self, mat2d):
        return sfcom.mat3d(mat2d, self.modelplane, reshape=True) # from common.py

    def evolve(self, u, S, dt, iota=+1, Gamma0=None, Lambda0=None, steadystate=False):
        """
        Evolve CPO using Laplacian stabilized, Euler time integration
        """
        if self.w is None:
            raise ValueError('CPO state "w" not set. Did you forget to initialize the CPO field?')

        self.w_prev.assign(self.w) # current state (w) must be set
        F = self.weakform(u, S, dt, iota, Gamma0, Lambda0, steadystate=steadystate)
        solve(lhs(F)==rhs(F), self.w, self.bcs, solver_parameters={'linear_solver':'gmres', }) # fastest tested are: gmres, bicgstab, tfqmr (For non-symmetric problems, a Krylov solver for non-symmetric systems, such as GMRES, is a better choice)
