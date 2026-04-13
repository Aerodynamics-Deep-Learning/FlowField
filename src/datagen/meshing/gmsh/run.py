import gmsh
import os

from .io import GMSH_Write_Exception
from .utils import get_mesh_height

from ...schemas import Airfoil
from .schemas import GMSH_In, GMSH_Out, GMSH_MeshingConfig, GMSH_ExitFlag

def GMSH_MeshGenerator(data: GMSH_In) -> GMSH_Out:

    """
    The actual function function that

    1- Initializes gmsh
    2- gets the dynamic y+ height, builds geom & mesh using the generate_mesh function
    3- validates and quality checks the mesh, based on element quality, determines the flag
    4- exportsthe final mesh with the .su2 extension (needs to build the path from working_dir)
    5- populate the verbose_list using functions in the io (the .geo dump done in this function, too heavy to move around the gmsh obj)
    6- return the whole thing as GMSH_Out

    if none of this works, 
    1- return the exception written, path included in the verbose list
    2- append the necessary flag
    3- return the whole thing as GMSH_Out

    !!! finally, use:
    
    '
    finally:
        gmsh.finalize()
    '
    """

    """
    Orchestrates the entire pipeline of mesh generation using GMSH, including flagging, io, etc.

    Args:
        data (GMSH_In): The input data schema defined for GMSH, includign the meshing configs, airfoil geometry, freeflow, and utils such as io
    
    Returns:
        GMSH_Out: The output data schema defined for GMSH, including the airfoil, freeflow, flag, mesh path, other stuff, and a verbose list
    """

    log_path = os.path.join(data.working_dir, f"{data.airfoil.airfoil_name}_gmsh_log.txt")
    geo_path = None # Fallback value, in case geo dump was not generated
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.logger.start(log_path) 
    except Exception:
        pass

    try:
        h_first = get_mesh_height(
            Re= data.freestream.Re,
            chord= data.airfoil.chord,
            target_yplus= data.meshing_config.target_yplus
        )

        geo_path = generate_mesh(
            meshing_config= data.meshing_config,
            airfoil= data.airfoil,
            h_first= h_first,
            working_dir=data.working_dir
        )

        # Flagging and metrics
        _, qualities = gmsh.model.mesh.getElementQuality(elementTags=[], qualityName="SICN")
        min_mesh_quality = min(qualities) if len(qualities) > 0 else 0.0
        num_nodes = len(gmsh.model.mesh.getNodes()[0])

        quad_tags, _ = gmsh.model.mesh.getElementsByType(3)
        num_quads = len(quad_tags)

        if num_quads == 0:
            flag = GMSH_ExitFlag.EXTRUSION_FAIL
        elif min_mesh_quality <= 0.1:
            flag = GMSH_ExitFlag.NEGATIVE_JACOBIAN
        else:
            flag = GMSH_ExitFlag.SUCCESS

        mesh_name = f"{data.airfoil.airfoil_name}_mesh.su2"
        mesh_path = os.path.join(data.working_dir, mesh_name)
        gmsh.write(mesh_path)

        # Build the verbose list with gmsh log path, .geo dump, and exception (None, because it is working as intended)
        verbose_list = [log_path, geo_path, None]

        return GMSH_Out(
           airfoil=data.airfoil,
           freestream=data.freestream,
           flag=flag,
           mesh_path=mesh_path,
           min_mesh_quality=min_mesh_quality,
           num_nodes=num_nodes,
           verbose_list=verbose_list 
        )

    except Exception as e:

        # Build the verbose list here
        exception_path = GMSH_Write_Exception(e, data.working_dir)
        verbose_list = [log_path, geo_path, exception_path]

        return GMSH_Out(
            airfoil=data.airfoil,
            freestream=data.freestream,
            flag=GMSH_ExitFlag.FATAL_ERROR,
            verbose_list=verbose_list
        )

    finally:
        try:
            gmsh.logger.stop()
        except Exception:
            pass
        gmsh.finalize()

def generate_mesh(meshing_config: GMSH_MeshingConfig, airfoil: Airfoil, h_first: float, working_dir: str) -> str:
    """
    Generates the mesh given the configs, airfoil geometry, and the first cell height

    Args:
        meshing_config (GMSH_MeshingConfig): The meshing config data schema
        airfoil (Airfoil): The airfoil geometry data schema
        h_first (float): The height of the first cell, derived with .utils.get_mesh_height
        working_dir (str): The string of the working directory

    Returns:
        None: A global api state of the generated mesh
        str: The path string representing the .geo dump
    """
    # Redefine for ease of use
    model = gmsh.model
    occ = model.occ

    # Individually add the points
    coords_numpy = airfoil.coords_tensor.numpy()
    point_tags = []
    for x, y in coords_numpy:
        point_tags.append(occ.addPoint(x, y, 0))

    # Infers the leading edge index, creates the spline and te line to close the geom
    le_idx = airfoil.le_idx
    upper_points = point_tags[:le_idx+1]
    upper_spline = occ.addSpline(upper_points)
    lower_points = point_tags[le_idx:]
    lower_spline = occ.addSpline(lower_points)
    te_line = occ.addLine(point_tags[0], point_tags[-1])
    airfoil_curves = [upper_spline, te_line, lower_spline]

    # Create the full airfoil boundary
    loop = occ.addCurveLoop(airfoil_curves)
    airfoil_surface = occ.addPlaneSurface([loop])

    # Getting the fluid domain and farfield disk
    farfield_disk = occ.addDisk(0.5, 0, 0, meshing_config.farfield_radius, meshing_config.farfield_radius)
    fluid_domain, _ = occ.cut([(2, farfield_disk)], [(2, airfoil_surface)])
    occ.synchronize()

    # Defining and creating all the tags for boundary, airfoil, and fluid 
    all_domain_boundaries = gmsh.model.getBoundary(fluid_domain, oriented=False)
    all_curve_tags = [tag for dim, tag in all_domain_boundaries]
    actual_airfoil_curves = []
    actual_farfield_curves = []

    for tag in all_curve_tags:
        bbox = gmsh.model.occ.getBoundingBox(1, tag)
        if abs(bbox[0]) > 5.0 or abs(bbox[3]) > 5.0: 
            actual_farfield_curves.append(tag)
        else:
            actual_airfoil_curves.append(tag)

    fluid_surface_tags = [tag for dim, tag in fluid_domain]

    
    # Defining boundary layer field
    field_id = model.mesh.field.add("BoundaryLayer")
    model.mesh.field.setNumbers(field_id, "CurvesList", actual_airfoil_curves)
    model.mesh.field.setNumber(field_id, "Size", h_first)
    model.mesh.field.setNumber(field_id, "Ratio", meshing_config.bl_growth_ratio)
    model.mesh.field.setNumber(field_id, "Quads", 1)
    model.mesh.field.setNumber(field_id, "Thickness", meshing_config.bl_total_thickness)
    model.mesh.field.setNumber(field_id, "FanNodes", meshing_config.bl_fan_elements)
    model.mesh.field.setAsBoundaryLayer(field_id)

    # Defining the groups
    gmsh.model.addPhysicalGroup(1, actual_airfoil_curves, name=meshing_config.marker_airfoil)
    gmsh.model.addPhysicalGroup(1, actual_farfield_curves, name=meshing_config.marker_farfield)
    gmsh.model.addPhysicalGroup(2, fluid_surface_tags, name=meshing_config.marker_fluid)

    # Additional definitions
    gmsh.option.setNumber("Mesh.Smoothing", meshing_config.smoothing_steps)
    gmsh.option.setNumber("Mesh.Algorithm", meshing_config.algorithm)

    # .geo dump, for diagnosis
    geo_path = os.path.join(working_dir, f"{airfoil.airfoil_name}.geo")
    gmsh.write(geo_path)

    model.mesh.generate(2)

    return geo_path