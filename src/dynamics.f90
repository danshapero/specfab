! N. M. Rathmann <rathmann@nbi.ku.dk>, 2019-2022

module dynamics  

    use tensorproducts
    use moments ! used by tensorial dynamics routines
    use gaunt

    implicit none 

    integer, parameter, private :: dp = 8 ! Default precision
    real,    parameter, private :: Pi = 3.141592653589793
    integer, parameter, private :: x = 1, y = 2, z = 3 ! Matrix indices (used for readability)
    complex(kind=dp), parameter, private :: r = (1,0), i = (0,1) ! real and imag units
    integer, private :: ii, ll, mm ! Loop indices

    integer, private   :: Lcap    ! Truncation "L" of expansion series (internal copy of what was passed to the init routine).
    integer, private   :: nlm_len ! Total number of expansion coefficients (i.e. DOFs)

    integer, parameter :: Lcap__max  = 30 ! Hard limit
    integer, parameter :: nlm_lenmax = (Lcap__max+1)*(Lcap__max+2)/2
    
    ! Static (constant) matrices used for spectral dynamics
    real(kind=dp), parameter :: Ldiag(nlm_lenmax) = [( (-ll*(ll+1),mm=-ll,ll), ll=0,  Lcap__max,2)] ! Diagonal entries of Laplacian diffusion operator.
    
contains      

    !---------------------------------
    ! INIT
    !---------------------------------

    subroutine initdynamics(Lcap_)

        ! Needs to be called once before using the module routines.

        implicit none    
        integer, intent(in) :: Lcap_ ! Truncation "Lcap"
        
        ! Save internal copy
        Lcap    = Lcap_ 
        nlm_len = (Lcap+1)*(Lcap+2)/2 

        ! Set gaunt coefficients (overlap integrals involving three spherical harmonics)
        call set_gaunts()
    end

    !---------------------------------
    ! FABRIC DYNAMICS
    !---------------------------------

    function dndt_ij_LATROT(eps,omg, tau,Aprime,Ecc,Eca, beta)  

        !------------------
        ! Lattice rotation
        !------------------

        ! Returns matrix dndt_ij such that d/dt (nlm)_i = dndt_ij (nlm)_j

        implicit none

        real(kind=dp), intent(in) :: eps(3,3), omg(3,3), tau(3,3) ! strain-rate (eps), spin (omg), dev. stress (tau)
        real(kind=dp), intent(in) :: Aprime, Ecc, Eca, beta
        complex(kind=dp)          :: dndt_ij_LATROT(nlm_len,nlm_len), qe(-2:2), qt(-2:2), qo(-1:1)
        integer, parameter        :: SHI_LATROT = 6 ! Scope of harmonic interactions (in wave space) for LATROT
        complex(kind=dp), dimension(SHI_LATROT) :: g0,     gz,     gn,     gp
        complex(kind=dp), dimension(SHI_LATROT) :: g0_rot, gz_rot, gn_rot, gp_rot
        complex(kind=dp), dimension(SHI_LATROT) :: g0_Sac, gz_Sac, gn_Sac, gp_Sac
        complex(kind=dp), dimension(SHI_LATROT) :: g0_Tay, gz_Tay, gn_Tay, gp_Tay
        real(kind=dp) :: etaprime

        ! Quadric expansion coefficients
        qe = quad_rr(eps)
        qt = quad_rr(tau)
        qo = quad_tp(omg)

        ! Harmonic interaction weights
        g0_rot = [ 0*r, 0*r,0*r,0*r,0*r,0*r ]
        gz_rot = [ -i*sqrt(3.)*qo(0), 0*r,0*r,0*r,0*r,0*r ]
        gn_rot = [ -i*6/sqrt(6.)*qo(-1), 0*r,0*r,0*r,0*r,0*r ]
        gp_rot = [ +i*6/sqrt(6.)*qo(+1), 0*r,0*r,0*r,0*r,0*r ]
        g0_Tay = 3*[ 0*r, qe(-2),qe(-1),qe(0),qe(+1),qe(+2) ]
        gz_Tay = [ 0*r, -qe(-2),0*r,0*r,0*r,qe(+2) ]
        gn_Tay = [ sqrt(5./6)*qe(-1), 0*r, qe(-2), sqrt(2./3)*qe(-1), sqrt(3./2)*qe(0), 2*qe(+1) ]
        gp_Tay = [ sqrt(5./6)*qe(+1), 2*qe(-1), sqrt(3./2)*qe(0), sqrt(2./3)*qe(+1), qe(+2), 0*r ]
        
        if (beta .gt. 1.0d0-1d-4) then
            ! beta=1 => ODF evolution is kinematic in the sense that c-axes rotate in response to the bulk stretching and spin.
            ! This is what was used in Rathmann et al. (2021) and Rathmann and Lilien (2021).
            
            g0 = g0_rot + g0_Tay
            gz = gz_rot + gz_Tay
            gn = gn_rot + gn_Tay
            gp = gp_rot + gp_Tay
        
        else
            ! *** EXPERIMENTAL, NOT VALIDATED ***
            ! Depends on (tau,Aprime,Ecc,Eca) unlike the Taylor case.
            
            etaprime = Aprime*doubleinner22(tau,tau) ! Assumes eta' = A'*(I2(tau)) (i.e. n'=3).
            g0_Sac = 3*[ 0*r, Eca*qt(-2),Eca*qt(-1),Eca*qt(0),Eca*qt(+1),Eca*qt(+2) ]
            gz_Sac = [ 0*r, -Eca*qt(-2),-1./2*(Eca-1)*qt(-1),0*r,1./2*(Eca-1)*qt(+1),Eca*qt(+2) ]
            gn_Sac = [ sqrt(5./6)*qt(-1), 0*r, Eca*qt(-2), sqrt(1./6)*(3*Eca-1)*qt(-1), sqrt(3./2)*Eca*qt(0), (Eca+1)*qt(+1) ]
            gp_Sac = [ sqrt(5./6)*qt(+1), (Eca+1)*qt(-1), sqrt(3./2)*Eca*qt(0), sqrt(1./6)*(3*Eca-1)*qt(+1), Eca*qt(+2), 0*r ]

            g0 = g0_rot + (1-beta)*etaprime*g0_Sac + beta*g0_Tay
            gz = gz_rot + (1-beta)*etaprime*gz_Sac + beta*gz_Tay
            gn = gn_rot + (1-beta)*etaprime*gn_Sac + beta*gn_Tay
            gp = gp_rot + (1-beta)*etaprime*gp_Sac + beta*gp_Tay
        end if 

        do ii = 1, nlm_len
            dndt_ij_LATROT(ii,1:nlm_len) = -1*( matmul(GC(ii,1:nlm_len,1:SHI_LATROT),g0) + matmul(GCm(ii,1:nlm_len,1:SHI_LATROT),gz) + matmul(GC_m1(ii,1:nlm_len,1:SHI_LATROT),gn) + matmul(GC_p1(ii,1:nlm_len,1:SHI_LATROT),gp) )    
        end do
    end

    function dndt_ij_DDRX(nlm, tau)

        !-----------------------------------------------
        ! Discontinuous dynamic recrystallization (DDRX)
        !----------------------------------------------- 
        
        ! Nucleation and migration recrystalization modeled as a decay process (Placidi et al., 2010).
        ! Returns matrix dndt_ij such that d/dt (nlm)_i = dndt_ij (nlm)_j
        ! NOTICE: This is Gamma/Gamma_0. The caller must multiply by an appropriate DDRX rate factor, Gamma_0(T,tau,eps,...).
        
        implicit none

        complex(kind=dp), intent(in) :: nlm(nlm_len)
        real(kind=dp), intent(in)    :: tau(3,3) ! Deviatoric stress tensor
        complex(kind=dp)             :: dndt_ij_DDRX(nlm_len,nlm_len)
        real(kind=dp)                :: Davg

        dndt_ij_DDRX = dndt_ij_DDRX_src(tau)

        ! Add add (nonlinear) sink term <D> to all diagonal entries such that dndt_ij = Gamma_ij/Gamma0 = (D_ij - <D>*I_ij)/(tau:tau) (where I is the identity)
        Davg = doubleinner22(matmul(tau,tau), a2(nlm)) - doubleinner22(tau,doubleinner42(a4(nlm),tau)) ! (tau.tau):a2 - tau:a4:tau 
        Davg = Davg/doubleinner22(tau,tau) ! normalize
        do ii = 1, nlm_len    
            dndt_ij_DDRX(ii,ii) = dndt_ij_DDRX(ii,ii) - Davg 
        end do
    end

    function dndt_ij_DDRX_src(tau)  

        !------------------
        ! DDRX source term
        !------------------

        implicit none

        real(kind=dp), intent(in) :: tau(3,3) ! dev. stress tensor
        complex(kind=dp)          :: dndt_ij_DDRX_src(nlm_len,nlm_len), qt(-2:2)
        integer, parameter        :: SHI_DDRX = 1+5+9 ! Scope of harmonic interactions (in wave space) for DDRX
        real(kind=dp)             :: k
        complex(kind=dp)          :: g(SHI_DDRX)

        ! Quadric expansion coefficients
        qt = quad_rr(tau)

        ! Harmonic interaction weights (requires qt, sets g and k)
        include "include/ddrx-coupling-weights.f90"

        ! D
        g = k*g ! common prefactor
        g = g/doubleinner22(tau,tau) ! normalize (can be done already here since it is a constant prefactor for dndt_ij_DDRX_src)
        do ii = 1, nlm_len    
            dndt_ij_DDRX_src(ii,1:nlm_len) = matmul(GC(ii,:nlm_len,:), g)
        end do
    end

    function dndt_ij_CDRX()

        !--------------------------------------------
        ! Continuous dynamic recrystalization (CDRX)
        !--------------------------------------------
        
        ! Rotation recrystalization (polygonization) as a Laplacian diffusion process (Godert, 2003).
        ! Returns matrix dndt_ij such that d/dt (nlm)_i = dndt_ij (nlm)_j
        ! NOTICE: This gives the unscaled effect of CDRX. The caller must multiply by an appropriate CDRX rate factor (scale) that should depend on temperature, stress, etc.

        implicit none
        real(kind=dp) :: dndt_ij_CDRX(nlm_len,nlm_len)

        dndt_ij_CDRX = 0.0
        do ii = 1, nlm_len 
            dndt_ij_CDRX(ii,ii) = Ldiag(ii) ! Laplacian
        end do  
    end

    function dndt_ij_REG(eps) 

        !----------------
        ! Regularization 
        !----------------
        
        ! Returns matrix dndt_ij such that d/dt (nlm)_i = dndt_ij (nlm)_j
        ! NOTICE: Calibrated for 4 <= L <= 8. For larger L, the caller  must specify an appropriate scaling.
        ! Calibration is provided by the script in tests/calibrate-regularization.
        
        implicit none
        real(kind=dp), intent(in) :: eps(3,3)
        real(kind=dp)             :: dndt_ij_REG(nlm_len,nlm_len)
        real(kind=dp)             :: nu, expo, scalefac
        
        if (Lcap == 4) then
            expo = 1.65
            nu   = 2.088913e+00
        end if 

        if (Lcap == 6) then
            expo = 2.35
            nu   = 2.914053e+00
        end if 
        
        if (Lcap == 8) then
            expo = 2.80
            nu   = 4.120216e+00
        end if 

        if (Lcap == 20) then
            expo = 3.00
            nu   = 2.007883e+01
        end if 

        if ((Lcap .gt. 8) .and. (Lcap .lt. 20)) then
    !        print *, 'specfab error: returning the unscaled (but normalized) Laplacian matrix for you to scale yourself.'
            expo = 1
            scalefac = -1
        else
            scalefac = -nu * norm2(reshape(eps,[size(eps)])) 
        end if
        
        dndt_ij_REG = 0.0
        do ii = 1, nlm_len 
            dndt_ij_REG(ii,ii) = scalefac * abs( Ldiag(ii)/(Lcap*(Lcap+1)) )**expo 
        end do
    end

    !---------------------------------
    ! FABRIC DYNAMICS IN TENSORIAL SPACE
    !---------------------------------

    include "tensorialdynamics.f90"

    !---------------------------------
    ! QUADRICS
    !---------------------------------

    function quad_rr(M) result (q2m)
        
        ! Expansion coefficients of "M : rr" assuming M is symmetric and r is the radial unit vector
        
        implicit none
        
        real(kind=dp), intent(in) :: M(3,3) 
        real(kind=dp), parameter :: fsq = sqrt(2*Pi/15) 
        complex(kind=dp) :: q2m(-2:2)
        
        ! components (2,-2), (2,-1), (2,0), (2,+1), (2,+2)
        q2m = [    +fsq*   (M(x,x)-M(y,y)+2*i*M(x,y)), &
                   +2*fsq* (M(x,z)+i*M(y,z)), &
                   -2./3*sqrt(Pi/5)*r*(M(x,x)+M(y,y)-2*M(z,z)), &
                   -2*fsq* (M(x,z)-i*M(y,z)), &
                   +fsq*   (M(x,x)-M(y,y)-2*i*M(x,y))]
    end
       
    function quad_tp(M) result (q1m)
        
        ! Expansion coefficients of "M : tp" assuming M is anti-symmetric and (t,p) are the (theta,phi) unit vectors.
        
        implicit none
        
        real(kind=dp), intent(in) :: M(3,3) 
        real(kind=dp), parameter :: fsq1 = sqrt(2*Pi/3)
        complex(kind=dp) :: q1m(-1:1)
        
        ! components (1,-1), (1,0), (1,+1)
        q1m = fsq1*[r*M(y,z)-i*M(x,z), r*sqrt(2.)*M(x,y), -r*M(y,z)-i*M(x,z)]
    end

end module dynamics