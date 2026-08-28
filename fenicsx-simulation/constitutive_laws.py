import ufl
import jax.numpy as jnp
import jax

def deformation_gradient(u):
    """Return deformation gradient F = I + grad(u)"""
    return ufl.Identity(len(u)) + ufl.grad(u)

def hencky_cauchy_stress_safe(F, mu, lamb):
    """
    Hencky (logarithmic) stress with safe branch for F ≈ I.
    """
    I = jnp.eye(3, dtype=F.dtype)
    # Check if F is close to identity (absolute tolerance)
    near_identity = jnp.allclose(F, I, atol=1e-8)

    def general_case(F):
        B = F @ jnp.swapaxes(F, -2, -1)
        eigenvals, eigenvecs = jnp.linalg.eigh(B)
        lambda_principal = jnp.sqrt(eigenvals)
        hencky_principal = jnp.log(lambda_principal)
        epsilon_H = eigenvecs @ (jnp.diag(hencky_principal) @ jnp.swapaxes(eigenvecs, -2, -1))
        trace_epsilon = jnp.trace(epsilon_H, axis1=-2, axis2=-1)
        I = jnp.eye(3)
        sigma = 2.0 * mu * epsilon_H + lamb * trace_epsilon[..., None, None] * I
        return sigma
    def linear_case(F):
        # Small strain approximation: ε = sym(F - I)
        epsilon = 0.5 * (F + F.swapaxes(-2, -1)) - I
        trace_eps = jnp.trace(epsilon, axis1=-2, axis2=-1)
        sigma = 2.0 * mu * epsilon + lamb * trace_eps[..., None, None] * I
        return sigma

    return jax.lax.cond(near_identity, linear_case, general_case, F)


def first_piola_kirchhoff(F, mu, lamb):
    """
    Compute First Piola-Kirchhoff stress from Cauchy stress.
    
    Args:
        F: Deformation gradient (3x3) or batch of (..., 3,3)
        sigma: Cauchy stress (3x3) or batch of (..., 3,3)
    
    Returns:
        P: First Piola-Kirchhoff stress (3x3)
    """
    J = jnp.linalg.det(F)
    F_inv_T = jnp.linalg.inv(F).swapaxes(-2, -1)  # F^{-T}

    sigma = hencky_cauchy_stress_safe(F,mu, lamb)
    piola = J[..., None, None] * sigma @ F_inv_T 
    state = piola,sigma
    return piola, state

def first_piola_kirchhoff_orthotropic(use_rotation=False):
    def func(F, C_voigt):
        """
        Compute First Piola-Kirchhoff stress from Cauchy stress.
        
        Args:
            F: Deformation gradient (3x3) or batch of (..., 3,3)
            sigma: Cauchy stress (3x3) or batch of (..., 3,3)
        
        Returns:
            P: First Piola-Kirchhoff stress (3x3)
        """
        J = jnp.linalg.det(F)
        F_inv_T = jnp.linalg.inv(F).swapaxes(-2, -1)  # F^{-T}

        sigma = hencky_cauchy_stress_orthotropic(F,C_voigt,use_rotation)
        piola = J[..., None, None] * sigma @ F_inv_T 
        state = piola,sigma
        return piola, state
    return func

def hencky_cauchy_stress_orthotropic(F, C_voigt, use_rotation= False):
    """
    Hencky stress with orthotropic stiffness C_voigt (6*6 NumPy array).
    """
    I = jnp.eye(3, dtype=F.dtype)
    near_identity = jnp.allclose(F, I, atol=1e-8)

    def general_case(F):
        # B = F @ jnp.swapaxes(F, -2, -1)
        # eigenvals, eigenvecs = jnp.linalg.eigh(B)
        # lambda_principal = jnp.sqrt(eigenvals)
        # hencky_principal = jnp.log(lambda_principal)
        # epsilon_H = eigenvecs @ (jnp.diag(hencky_principal) @ jnp.swapaxes(eigenvecs, -2, -1))
        # eps_voigt = strain_to_voigt(epsilon_H)
        # sigma_voigt = C_voigt @ eps_voigt
        # return voigt_to_stress(sigma_voigt)
        B = F @ F.T
        eigenvals, eigenvecs = jnp.linalg.eigh(B)
        lambda_principal = jnp.sqrt(eigenvals)
        hencky_principal = jnp.log(lambda_principal)
        eps_spatial = eigenvecs @ jnp.diag(hencky_principal) @ eigenvecs.T
        
        if use_rotation:
            # Orthotropic: rotate to material frame, apply C, rotate back
            U_svd, s, Vh = jnp.linalg.svd(F, full_matrices=False)
            R = U_svd @ Vh
            eps_material = R.T @ eps_spatial @ R
            eps_voigt = strain_to_voigt(eps_material)
            sigma_voigt = C_voigt @ eps_voigt
            sigma_material = voigt_to_stress(sigma_voigt)
            return R @ sigma_material @ R.T
        else:
            # Isotropic: apply directly in spatial frame
            eps_voigt = strain_to_voigt(eps_spatial)
            sigma_voigt = C_voigt @ eps_voigt
            return voigt_to_stress(sigma_voigt)

    def linear_case(F):
        epsilon = 0.5 * (F + F.swapaxes(-2, -1)) - I
        eps_voigt = strain_to_voigt(epsilon)
        sigma_voigt = C_voigt @ eps_voigt
        return voigt_to_stress(sigma_voigt)

    return jax.lax.cond(near_identity, linear_case, general_case, F)

def strain_to_voigt(eps):
    """3*3 symmetric strain → 6*1 Voigt vector"""
    return jnp.array([eps[0,0], eps[1,1], eps[2,2],
                      2*eps[0,1], 2*eps[0,2], 2*eps[1,2]])
def voigt_to_stress(s_voigt):
    """6*1 Voigt vector → 3*3 symmetric stress"""
    return jnp.array([[s_voigt[0], s_voigt[3], s_voigt[4]],
                      [s_voigt[3], s_voigt[1], s_voigt[5]],
                      [s_voigt[4], s_voigt[5], s_voigt[2]]])