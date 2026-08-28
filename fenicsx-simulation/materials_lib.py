from materials import IsotropicMaterial, OrthotropicMaterial

# Units: Pa, kg/m^3

# Ceramic
Alumina = IsotropicMaterial(
    young_modulus=320e9,
    rho=3700,
    poisson_ratio=0.23,
    name='alumina'
)

Sil_Carbide = IsotropicMaterial(
    young_modulus=410e9,
    rho=3163,
    poisson_ratio=0.21,
    name = "sil_carbide"
)

Bor_Carbide = IsotropicMaterial(
    young_modulus=430e9,
    rho=2510,
    poisson_ratio=0.18,
    name= 'bor_carbide'
)

# Backplate
AR500 = IsotropicMaterial(
    young_modulus=210e9,
    rho=7860,
    poisson_ratio=0.33,
    name = 'ar500'
)

Titanium_alloy = IsotropicMaterial(
    young_modulus=114e9,
    rho=4428,
    poisson_ratio=0.342,
    name= 'titanium_alloy'
)

# Kevlars / Aramid woven fabrics
# Convention:
# 1 = warp direction, in-plane
# 2 = weft/fill direction, in-plane
# 3 = through-thickness direction

Kevlar29 = OrthotropicMaterial(
    e11=62e9,
    e22=62e9,
    e33=2e9,
    nu12=0.35,
    nu13=0.30,
    nu23=0.30,
    g12=0.7e9,
    g13=0.6e9,
    g23=0.6e9,
    rho=1400,
    name= 'kevlar29'
)

Kevlar49 = OrthotropicMaterial(
    e11=151.7e9,
    e22=151.7e9,
    e33=4.14e9,
    nu12=0.35,
    nu13=0.30,
    nu23=0.30,
    g12=0.53e9,
    g13=0.53e9,
    g23=0.53e9,
    rho=1467,
    name = 'kevlar49'
)

Twaron = OrthotropicMaterial(
    e11=90e9,
    e22=90e9,
    e33=1.6e9,
    nu12=0.35,
    nu13=0.30,
    nu23=0.30,
    g12=3.6e9,
    g13=3.6e9,
    g23=3.6e9,
    rho=1450,
    name = 'twaron'
)