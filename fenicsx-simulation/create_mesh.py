import gmsh
import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from dolfinx import mesh


def create_mesh(width=0.4, height = 0.4, thickness = 0.010, mesh_size= 0.01, verbose = False):
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 1) 
    plate = gmsh.model.occ.add_box(0,0,0,width,height,thickness)
    gmsh.model.occ.synchronize()

    gdim = 3

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(surface[1])
    gmsh.model.mesh.setTransfiniteVolume(plate)

    for surface in surfaces:
        gmsh.model.mesh.setRecombine(2, surface[1])

    gmsh.model.addPhysicalGroup(gdim, [plate], 1)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(gdim)



    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim, partitioner=partitioner)
    assert mesh_data.cell_tags is not None
    cell_markers = mesh_data.cell_tags
    domain = mesh_data.mesh

    gmsh.finalize()
    return domain

# def create_mesh(width=0.4, height=0.4, thickness=0.010, mesh_size=0.01, n_elements_thickness=5):
#     gmsh.initialize()
    
#     # Create 3D box directly but with transfinite meshing
#     box = gmsh.model.occ.add_box(0, 0, 0, width, height, thickness)
#     gmsh.model.occ.synchronize()
    
#     # Get all entities
#     volumes = gmsh.model.getEntities(3)
#     surfaces = gmsh.model.getEntities(2)
#     edges = gmsh.model.getEntities(1)
    
#     # Calculate number of elements in x and y directions based on mesh_size
#     nx = max(1, int(width / mesh_size))
#     ny = max(1, int(height / mesh_size))
    
#     # Set transfinite curves for edges
#     # Group edges by their orientation
#     x_edges = []
#     y_edges = []
#     z_edges = []
    
#     for edge in edges:
#         # Get bounding box of the edge
#         min_x, min_y, min_z, max_x, max_y, max_z = gmsh.model.getBoundingBox(1, edge[1])
#         dx = max_x - min_x
#         dy = max_y - min_y
#         dz = max_z - min_z
        
#         if dx > 1e-6 and dy < 1e-6 and dz < 1e-6:
#             # Edge along x-direction
#             x_edges.append(edge[1])
#         elif dy > 1e-6 and dx < 1e-6 and dz < 1e-6:
#             # Edge along y-direction
#             y_edges.append(edge[1])
#         elif dz > 1e-6 and dx < 1e-6 and dy < 1e-6:
#             # Edge along z-direction (thickness)
#             z_edges.append(edge[1])
    
#     # Set number of elements on each edge
#     for edge_tag in x_edges:
#         gmsh.model.mesh.setTransfiniteCurve(edge_tag, nx)
#     for edge_tag in y_edges:
#         gmsh.model.mesh.setTransfiniteCurve(edge_tag, ny)
#     for edge_tag in z_edges:
#         gmsh.model.mesh.setTransfiniteCurve(edge_tag, n_elements_thickness)
    
#     # Set transfinite surfaces
#     for surface in surfaces:
#         gmsh.model.mesh.setTransfiniteSurface(surface[1], "Left")
    
#     # Set transfinite volume
#     for volume in volumes:
#         gmsh.model.mesh.setTransfiniteVolume(volume[1])
    
#     # Recombine to get hexahedra
#     for surface in surfaces:
#         gmsh.model.mesh.setRecombine(2, surface[1])
    
#     # Physical group
#     gmsh.model.addPhysicalGroup(3, [volumes[0][1]], 1)
    
#     # Generate mesh
#     gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
#     gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
#     gmsh.model.mesh.generate(3)
    
#     # Import to DOLFINx
#     gmsh_model_rank = 0
#     mesh_comm = MPI.COMM_WORLD
#     partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
#     mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=3, partitioner=partitioner)
    
#     gmsh.finalize()
#     return mesh_data.mesh