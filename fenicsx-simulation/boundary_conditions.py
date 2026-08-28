import numpy as np
from dolfinx import mesh, fem, default_scalar_type


def clamped_boundary(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0),np.isclose(x[0], 0.4)),
        np.logical_or(np.isclose(x[1], 0),np.isclose(x[1], 0.4))
        )

def clamped_boundary_condition(V, domain)-> fem.DirichletBC:
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    return bc

def point_z_boundary_condition(point, V)-> fem.DirichletBC:
    def locate_center(x):
        return np.isclose(x[0], point[0], atol=1e-6) & \
               np.isclose(x[1], point[1], atol=1e-6) & \
               np.isclose(x[2], point[2], atol=1e-6)
    point_dofs = fem.locate_dofs_geometrical(V, locate_center)
    if len(point_dofs) == 0:
        print(f"ERROR: Center point {point} not found!")
        return None
    u_D_center = np.array([0., 0., -0.05], dtype=default_scalar_type)
    center_bc = fem.dirichletbc(u_D_center, point_dofs, V)

def create_circular_boundary_condition(V, domain, thickness=0.01, center_x=0.2, center_y=0.2, radius=0.05, displacement=-0.05):
    """
    Apply displacement to a circular region on top surface of the plate.
    """
    
    def on_top_surface_and_in_circle(x):
        """Find top surface points inside the circle."""
        # Check if on top surface (tolerance for floating point)
        on_top = np.isclose(x[2,:], thickness, atol=1e-6)
        # Check if within circle radius
        in_circle = (x[0,:] - center_x)**2 + (x[1,:] - center_y)**2 < radius**2
        # for i in range(len(in_circle)):
        #     if in_circle[i]:
        #         print(x[:,i])
        #         print(on_top[i])
        return on_top & in_circle
    
    # Locate facets (2D entities) on top surface within circle
    vertices = mesh.locate_entities(domain, 0, on_top_surface_and_in_circle)
    
    # Get DOFs on these facets for the z-component of displacement
    vertices_dofs = fem.locate_dofs_topological(V.sub(2), 0, vertices)
    # print(vertices_dofs)
    # Create constant displacement value
    u_D_center = fem.Constant(domain, default_scalar_type(displacement))
    
    # Create Dirichlet BC
    bc = fem.dirichletbc(u_D_center, vertices_dofs, V.sub(2))
    
    return bc, u_D_center
