from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from enum import Enum, auto


class MaterialType(Enum):
    Isotropic= auto()
    Orthotropic = auto()

@dataclass(slots=True)
class IsotropicMaterial:
    young_modulus: float
    poisson_ratio: float
    rho: float
    name: str
    material_type: MaterialType = MaterialType.Isotropic

    @property
    def mu(self) -> float:
        return self.young_modulus / (2 * (1 + self.poisson_ratio))
    
    @property
    def lamb(self) -> float:
        return (self.young_modulus * self.poisson_ratio) / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
    
    @property
    def elasticity_matrix(self) -> npt.NDArray[np.float64]:
        """6*6 Voigt stiffness matrix for isotropic material."""
        mu = self.mu
        lamb = self.lamb
        C = np.zeros((6, 6), dtype=np.float64)
        
        # Normal components
        C[0, 0] = C[1, 1] = C[2, 2] = lamb + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lamb
        
        # Shear components
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        
        return C


@dataclass(slots=True)
class OrthotropicMaterial:
    e11: float  # Young's modulus in 1-direction (warp)
    e22: float  # Young's modulus in 2-direction (weft)
    e33: float  # Young's modulus in 3-direction (through-thickness)
    nu12: float
    nu13: float
    nu23: float
    g12: float  # Shear modulus in 1-2 plane
    g13: float  # Shear modulus in 1-3 plane
    g23: float  # Shear modulus in 2-3 plane
    rho: float

    name: str
    material_type: MaterialType = MaterialType.Orthotropic
    
    @property
    def elasticity_matrix(self) -> npt.NDArray[np.float64]:
        """6*6 Voigt stiffness matrix for orthotropic material."""
        # Build compliance matrix first
        S = np.zeros((6, 6), dtype=np.float64)
        S[0, 0] = 1.0 / self.e11
        S[1, 1] = 1.0 / self.e22
        S[2, 2] = 1.0 / self.e33
        
        S[0, 1] = S[1, 0] = -self.nu12 / self.e11
        S[0, 2] = S[2, 0] = -self.nu13 / self.e11
        S[1, 2] = S[2, 1] = -self.nu23 / self.e22
        
        S[3, 3] = 1.0 / self.g12
        S[4, 4] = 1.0 / self.g13
        S[5, 5] = 1.0 / self.g23
        
        # Invert to get stiffness
        C = np.linalg.inv(S)
        return C

