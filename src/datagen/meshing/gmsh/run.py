import gmsh
import os
import numpy as np
import math

from .io import GMSH_Write_Exception
from .utils import GMSH_get_mesh_height, GMSH_export_sicn_histogram

from ...schemas import Airfoil
from .schemas import GMSH_In, GMSH_Out, GMSH_MeshingConfig, GMSH_ExitFlag

import logging
logger = logging.getLogger(__name__)

def GMSH_MeshGenerator(data: GMSH_In) -> GMSH_Out:
    """
    Orchestrates the entire pipeline of mesh generation using GMSH, including flagging, io, etc.

    Args:
        data (GMSH_In): The input data schema defined for GMSH, includign the meshing configs, airfoil geometry, freeflow, and utils such as io
    
    Returns:
        GMSH_Out: The output data schema defined for GMSH, including the airfoil, freeflow, flag, mesh path, other stuff, and a verbose list
    """

    log_path = os.path.join(data.working_dir, f"{data.airfoil.airfoil_name}_gmsh_log.txt")
    brep_path = None # Fallback value, in case brep dump was not generated
    
    gmsh.initialize()
    gmsh.model.add("airfoil_hybrid_cmesh")

    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.logger.start() 

    try:
        h_first = GMSH_get_mesh_height(
            Re= data.freestream.Re,
            chord= data.airfoil.chord,
            target_yplus= data.meshing_config.target_yplus
        )

        # Checks the trailing edge length, if it is too small or blunted at all
        te_gap = np.linalg.norm(data.airfoil.coords_tensor[0] - data.airfoil.coords_tensor[-1])
        if te_gap == 0:
            logger.warning(f"Airfoil appears to not have been blunted. The first and last points are coincident at {data.airfoil.coords_tensor[0]}, this may cause problems with meshing")
            return GMSH_Out(
                airfoil=data.airfoil,
                freestream=data.freestream,
                flag=GMSH_ExitFlag.BLUNTING_FAIL,
                verbose_list=[log_path, None, None]
            )
        elif te_gap < 1e-5:
            logger.warning(f"Trailing edge gap is small: {te_gap}, this will most likely cause problems with TE meshing")
            return GMSH_Out(
                airfoil=data.airfoil,
                freestream=data.freestream,
                flag=GMSH_ExitFlag.BLUNTING_FAIL,
                verbose_list=[log_path, None, None]
            )
        else:
            pass

        # Get the mesh
        brep_path = generate_cmesh(
            meshing_config= data.meshing_config,
            airfoil= data.airfoil,
            h_first= h_first,
            working_dir=data.working_dir
        )

        # Flagging and metrics
        min_mesh_quality = 0.0
        # Extract tags for 2D traingles and quads
        tri_tags, _ = gmsh.model.mesh.getElementsByType(2)
        quad_tags, _ = gmsh.model.mesh.getElementsByType(3)
        all_2d_tags = np.concatenate([tri_tags, quad_tags]).astype(np.uint64)
        if len(all_2d_tags) > 0:
            qualities = gmsh.model.mesh.getElementQualities(elementTags=all_2d_tags, qualityName="minSICN")
            min_mesh_quality = min(qualities)
            print(f"Min mesh quality: {min_mesh_quality}")
            hist_path = os.path.join(data.working_dir, f"{data.airfoil.airfoil_name}_sicn_hist.png")
            GMSH_export_sicn_histogram(qualities, hist_path)
        else:
            flag = GMSH_ExitFlag.EXTRUSION_FAIL

        num_nodes = len(gmsh.model.mesh.getNodes()[0])
        num_quads = len(quad_tags)
        print(f"Num nodes: {num_nodes}, num quads: {num_quads}")

        if (num_quads == 0):
            flag = GMSH_ExitFlag.EXTRUSION_FAIL
        elif min_mesh_quality <= 0.0:
            flag = GMSH_ExitFlag.NEGATIVE_JACOBIAN
        else:
            flag = GMSH_ExitFlag.SUCCESS

        gmsh.option.setNumber("Mesh.SaveAll", 0)
        mesh_name = f"{data.airfoil.airfoil_name}_mesh.su2"
        mesh_path = os.path.join(data.working_dir, mesh_name)
        gmsh.write(mesh_path)

        mesh_name_vtk = f"{data.airfoil.airfoil_name}_mesh.vtk"
        mesh_path_vtk = os.path.join(data.working_dir, mesh_name_vtk)
        gmsh.write(mesh_path_vtk)

        # Build the verbose list with gmsh log path, .brep dump, and exception (None, because it is working as intended)
        verbose_list = [log_path, brep_path, None]

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
        verbose_list = [log_path, brep_path, exception_path]

        return GMSH_Out(
            airfoil=data.airfoil,
            freestream=data.freestream,
            flag=GMSH_ExitFlag.FATAL_ERROR,
            verbose_list=verbose_list
        )

    finally:
        try:
            log_messages = gmsh.logger.get()
            with open(log_path, "w") as log_file:
                for msg in log_messages:
                    log_file.write(f"{msg}\n")
            gmsh.logger.stop()
        except Exception:
            pass
        gmsh.finalize()

def generate_cmesh(meshing_config: GMSH_MeshingConfig, airfoil: Airfoil, h_first: float, working_dir: str) -> str:
    """
    Generates a C-mesh given the configs, airfoil geometry, and the first cell height

    Args:
        meshing_config (GMSH_MeshingConfig): The meshing config data schema
        airfoil (Airfoil): The airfoil geometry data schema
        h_first (float): The height of the first cell, derived with .utils.get_mesh_height
        working_dir (str): The string of the working directory

    Returns:
        None: A global api state of the generated mesh
        str: The path string representing the .brep dump
    """

    occ = gmsh.model.occ
    mesh = gmsh.model.mesh

    #region Inputs and Initials
    ## 1- INFER DATA, UNPACK ##
    # 1a- Infer data from the data agreements #
    # Infer the distance configs through the meshing_config data agreement
    farfield_radius = meshing_config.farfield_radius
    bl_thickness = meshing_config.bl_thickness
    wake_length = meshing_config.wake_length
    # Boundary layer meshing configs
    nx_le1 = meshing_config.nx_le1 # x-axis discretization for upper le
    nx_le2 = meshing_config.nx_le2 # x-axis discretization for lower le
    nx_upper = meshing_config.nx_upper # x-axis disc. for upper airfoil
    nx_lower = meshing_config.nx_lower # x-axis disc. for lower airfoil
    nx_wake = meshing_config.nx_wake # x-axis dics. for the wake
    bl_growth_ratio = meshing_config.bl_growth_ratio # Growth ratio for the y-axis
    bl_growth_ratio_inv = 1.0 / bl_growth_ratio
    wake_progression = meshing_config.wake_progression # wake progression ratio for the x-axis in wake
    wake_progression_inv = 1.0 / wake_progression
    chord_bump = meshing_config.chord_bump # The chord bump
    # Farfield layer meshing configs
    ff_growth_ratio = meshing_config.ff_growth_ratio
    ff_growth_ratio_inv = 1.0 / ff_growth_ratio

    # 1b- Unpack the coords #
    coords_tensor = airfoil.coords_tensor # This is [te_upper -> ... upper_anchor ... -> le -> ... lower_anhor ... -> te_lower]
    # Unpack indices
    le_idx = airfoil.le_idx
    upper_anchor_idx = meshing_config.upper_anchor_idx
    lower_anchor_idx = meshing_config.lower_anchor_idx
    # Unpack coords of LE
    le_x = coords_tensor[le_idx, 0].item()
    le_y = coords_tensor[le_idx, 1].item()
    te_upper_x = coords_tensor[0, 0].item()
    te_upper_y = coords_tensor[0, 1].item()
    te_lower_x = coords_tensor[-1, 0].item()
    te_lower_y = coords_tensor[-1, 1].item()

    # 1c- Dynamic n_by/te/farfield calculation #
    # Boundary layer
    n_cells_bl = math.log(1.0 + (bl_thickness / h_first) * (bl_growth_ratio - 1.0)) / math.log(bl_growth_ratio)
    ny_bl = math.ceil(n_cells_bl) + 1
    h_last_bl = h_first * (bl_growth_ratio ** (ny_bl - 2))
    h_first_ff = h_last_bl
    # Farfield
    ff_length = farfield_radius - bl_thickness
    n_cells_ff = math.log(1.0 + (ff_length / h_first_ff) * (ff_growth_ratio - 1.0)) / math.log(ff_growth_ratio)
    ny_farfield = math.ceil(n_cells_ff) + 1
    # TE
    te_thickness = te_upper_y - te_lower_y
    te_coarsen_factor = meshing_config.te_coarsen_factor
    target_h_te = h_first * te_coarsen_factor
    n_cells_te = te_thickness / target_h_te
    ny_te = max(math.ceil(n_cells_te) + 1, 3) # Ensures at least 2 cells

    ## 2- AIRFOIL RELATED CURVES ##
    # All airfoil points, in CCW direction [te_upper -> ... upper_anchor ... -> le -> ... lower_anhor ... -> te_lower]
    all_airfoil_points = []
    for pt in coords_tensor.numpy():
        all_airfoil_points.append(occ.addPoint(pt[0], pt[1], 0.0))
    # Anchor/LE/TE points
    le_point = all_airfoil_points[le_idx]
    anchor_upper_point = all_airfoil_points[upper_anchor_idx]
    anchor_lower_point = all_airfoil_points[lower_anchor_idx]
    te_upper_point = all_airfoil_points[0]
    te_lower_point = all_airfoil_points[-1]
    # Upper points
    upper_points = all_airfoil_points[:upper_anchor_idx+1][::-1] # This is [upper_anchor -> upper -> te_upper], CW
    # LE points (1 is upper, 2 is lower leading edge)
    le1_points = all_airfoil_points[upper_anchor_idx:le_idx+1][::-1] # This is [le -> upper_anchor], CW
    le2_points = all_airfoil_points[le_idx:lower_anchor_idx+1][::-1] # This is [lower_anchor -> le ], CW
    # Lower points
    lower_points = all_airfoil_points[lower_anchor_idx:][::-1] # This is [te_lower -> lower -> lower_anchor], CW
    # Creates the curves and blunt line in CW direction
    upper_curve = occ.addSpline(upper_points) # This is [upper_anchor-> upper -> te_upper], CW
    le1_curve = occ.addSpline(le1_points) # This is [le -> upper_anchor], CW
    le2_curve = occ.addSpline(le2_points) # This is [lower_anchor -> le], CW
    lower_curve = occ.addSpline(lower_points) # This is [te_lower -> lower -> lower_anchor], CW
    te_blunt_line = occ.addLine(te_upper_point, te_lower_point) # This is [te_upper, te_lower], CW
    #endregion


    #region Point, Line, Loop, Surface Defs
    ## 3- DEFINE POINTS
    # 3a- DEFINE BL/WAKE POINTS #
    # BL_UP_xx_P (Boundary line upper, right above the airfoil) xx = le (leading edge) or te (trailing edge)
    bl_up_le_p = occ.addPoint(le_x, le_y + bl_thickness, 0.0)
    bl_up_te_p = occ.addPoint(te_upper_x, le_y + bl_thickness, 0.0)
    # BL_LO_xx_P (Boundary line lower, right below the airfoil)
    bl_lo_le_p = occ.addPoint(le_x, le_y - bl_thickness, 0.0)
    bl_lo_te_p = occ.addPoint(te_lower_x, le_y - bl_thickness, 0.0)
    # BL_WT_x_p (Boundary line, trail of airfoil) x = u (upper) or l (lower)
    bl_wt_up_p = occ.addPoint(te_upper_x + wake_length, te_upper_y, 0.0)
    bl_wt_lo_p = occ.addPoint(te_lower_x + wake_length, te_lower_y, 0.0)
    # BL_Wx_p (Boundary line, wake of airfoil) x = u (upper) or l (lower)
    bl_wu_p = occ.addPoint(te_upper_x + wake_length, te_upper_y + bl_thickness, 0.0)
    bl_wl_p = occ.addPoint(te_lower_x + wake_length, te_lower_y - bl_thickness, 0.0)
    # BL_LE_P (Boundary line, left of the airfoil's LE)
    bl_le_p = occ.addPoint(le_x - bl_thickness, le_y, 0.0)

    # 3b- DEFINE FF POINTS #
    # FF_xx_up/ex_P (Farfield, above the airfoil far away) xx = le (above leading edge) or te (above trailing edge) or wu (above upper wake), up/ex (upper or exhaust)
    ff_up_le_p = occ.addPoint(le_x, le_y + farfield_radius, 0.0)
    ff_up_te_p = occ.addPoint(te_upper_x, le_y + farfield_radius, 0.0)
    ff_wu_ex_p = occ.addPoint(te_upper_x + wake_length, le_y + farfield_radius, 0.0)
    # FF_xx_l/e_P (Farfield, below the airfoil far away) xx = le (below leading edge) or te (above trailing edge) or wu (below lower wake), u/e (lower or exhaust)
    ff_lo_le_p = occ.addPoint(le_x, le_y - farfield_radius, 0.0)
    ff_lo_te_p = occ.addPoint(te_lower_x, le_y - farfield_radius, 0.0)
    ff_wl_ex_p = occ.addPoint(te_lower_x + wake_length, le_y - farfield_radius, 0.0)
    # FF_LE_P (Farfield, left of airfoil's LE)
    ff_le_p = occ.addPoint(le_x - farfield_radius, le_y, 0.0)


    ## 4- DEFINE LINES, CURVES, PLANES ##
    # 4a- DEFINE BL/WAKE LINES, CURVES, PLANES ##
    # BL_UP
    bl_up_te_l = occ.addLine(te_upper_point, bl_up_te_p) # CCW
    bl_up_up_l = occ.addLine(bl_up_te_p, bl_up_le_p) # CCW
    bl_up_le_l = occ.addLine(bl_up_le_p, anchor_upper_point) # CCW
    bl_up_curve = occ.addCurveLoop([upper_curve, bl_up_te_l, bl_up_up_l, bl_up_le_l]) # CCW
    bl_up_surface = occ.addPlaneSurface([bl_up_curve]) # CCW, FLUID
    # BL_LO
    bl_lo_te_l = occ.addLine(bl_lo_te_p, te_lower_point) # CCW
    bl_lo_le_l = occ.addLine(anchor_lower_point, bl_lo_le_p) # CCW
    bl_lo_lo_l = occ.addLine(bl_lo_le_p, bl_lo_te_p) # CCW
    bl_lo_curve = occ.addCurveLoop([lower_curve, bl_lo_le_l, bl_lo_lo_l, bl_lo_te_l]) # CCW
    bl_lo_surface = occ.addPlaneSurface([bl_lo_curve]) # CCW, FLUID
    # BL_WT
    bl_wt_up_l = occ.addLine(bl_wt_up_p, te_upper_point) # CCW
    bl_wt_lo_l = occ.addLine(te_lower_point, bl_wt_lo_p) # CCW
    bl_wt_ex_l = occ.addLine(bl_wt_lo_p, bl_wt_up_p) # CCW
    bl_wt_curve = occ.addCurveLoop([bl_wt_up_l, te_blunt_line, bl_wt_lo_l, bl_wt_ex_l]) # CCW
    bl_wt_surface = occ.addPlaneSurface([bl_wt_curve]) # CCW, FLUID
    # BL_WU
    bl_wu_ex_l = occ.addLine(bl_wt_up_p, bl_wu_p) # CCW
    bl_wu_up_l = occ.addLine(bl_wu_p, bl_up_te_p) # CCW
    bl_wu_curve = occ.addCurveLoop([bl_wu_ex_l, bl_wu_up_l, -bl_up_te_l, -bl_wt_up_l]) # CCW
    bl_wu_surface = occ.addPlaneSurface([bl_wu_curve]) # CCW, FLUID
    # BL_W
    bl_wl_lo_l = occ.addLine(bl_lo_te_p, bl_wl_p) # CCW
    bl_wl_ex_l = occ.addLine(bl_wl_p, bl_wt_lo_p) # CCW
    bl_wl_curve = occ.addCurveLoop([bl_wl_lo_l, bl_wl_ex_l, -bl_wt_lo_l, -bl_lo_te_l]) # CCW
    bl_wl_surface = occ.addPlaneSurface([bl_wl_curve]) # CCW, FLUID
    # BL_LE1
    bl_le1_l = occ.addCircleArc(bl_up_le_p, le_point, bl_le_p) # CCW
    bl_le_l = occ.addLine(bl_le_p, le_point) # CCW
    bl_le1_curve = occ.addCurveLoop([bl_le1_l, bl_le_l, le1_curve, -bl_up_le_l]) # CCW
    bl_le1_surface = occ.addPlaneSurface([bl_le1_curve]) # CCW, FLUID
    # BL_LE2
    bl_le2_l = occ.addCircleArc(bl_le_p, le_point, bl_lo_le_p) # CCW
    bl_le2_curve = occ.addCurveLoop([bl_le2_l, -bl_lo_le_l, le2_curve, -bl_le_l]) # CCW
    bl_le2_surface = occ.addPlaneSurface([bl_le2_curve]) # CCW, FLUID

    # 4b- DEFINE FF LINES, CURVES, PLANES #
    # FF_UP
    ff_up_te_l = occ.addLine(bl_up_te_p, ff_up_te_p) # CCW
    ff_up_up_l = occ.addLine(ff_up_te_p, ff_up_le_p) # CCW
    ff_up_le_l = occ.addLine(ff_up_le_p, bl_up_le_p) # CCW
    ff_up_curve = occ.addCurveLoop([ff_up_te_l, ff_up_up_l, ff_up_le_l, -bl_up_up_l]) # CCW
    ff_up_surface = occ.addPlaneSurface([ff_up_curve]) # CCW, FLUID
    # FF_LO
    ff_lo_le_l = occ.addLine(bl_lo_le_p, ff_lo_le_p) # CCW
    ff_lo_lo_l = occ.addLine(ff_lo_le_p, ff_lo_te_p) # CCW
    ff_lo_te_l = occ.addLine(ff_lo_te_p, bl_lo_te_p) # CCW
    ff_lo_curve = occ.addCurveLoop([ff_lo_le_l, ff_lo_lo_l, ff_lo_te_l, -bl_lo_lo_l]) # CCW
    ff_lo_surface = occ.addPlaneSurface([ff_lo_curve]) # CCW, FLUID
    # FF_WU
    ff_wu_ex_l = occ.addLine(bl_wu_p, ff_wu_ex_p) # CCW
    ff_wu_up_l = occ.addLine(ff_wu_ex_p, ff_up_te_p) # CCW
    ff_wu_curve = occ.addCurveLoop([ff_wu_ex_l, ff_wu_up_l, -ff_up_te_l, -bl_wu_up_l]) # CCW
    ff_wu_surface = occ.addPlaneSurface([ff_wu_curve]) # CCW, FLUID
    # FF_WL
    ff_wl_lo_l = occ.addLine(ff_lo_te_p, ff_wl_ex_p) # CCW
    ff_wl_ex_l = occ.addLine(ff_wl_ex_p, bl_wl_p) # CCW
    ff_wl_curve = occ.addCurveLoop([ff_wl_lo_l, ff_wl_ex_l, -bl_wl_lo_l, -ff_lo_te_l]) # CCW
    ff_wl_surface = occ.addPlaneSurface([ff_wl_curve]) # CCW, FLUID
    # FF_LE1
    ff_le1_l = occ.addCircleArc(ff_up_le_p, le_point, ff_le_p) # CCW
    ff_le_l = occ.addLine(ff_le_p, bl_le_p) # CCW
    ff_le1_curve = occ.addCurveLoop([ff_le1_l, ff_le_l, -bl_le1_l, -ff_up_le_l]) # CCW
    ff_le1_surface = occ.addPlaneSurface([ff_le1_curve]) # CCW, FLUID
    # FF_LE2
    ff_le2_l = occ.addCircleArc(ff_le_p, le_point, ff_lo_le_p) # CCW
    ff_le2_curve = occ.addCurveLoop([ff_le2_l, -ff_lo_le_l, -bl_le2_l, -ff_le_l]) # CCW
    ff_le2_surface = occ.addPlaneSurface([ff_le2_curve]) # CCW, FLUID
    #endregion


    #region Meshing Configuratins
    # Synchronize kernel first
    occ.synchronize()
    ## 5- TRANSFINITE MESHING FOR BOUNDARY LAYER REGIONS ##
    # 5a- VERTICAL PROGRESSION #
    bl_radial_outward = [ # Lines drawn OUTWARD from the wall (use normal growth)
        bl_up_te_l, # TE upper to BL edge
        bl_lo_le_l, # Lower anchor to BL edge
        bl_wu_ex_l  # Wake core edge to upper wake edge
    ]
    for tag in bl_radial_outward:
        mesh.setTransfiniteCurve(tag, ny_bl, "Progression", bl_growth_ratio)
    bl_radial_inward = [ # Lines drawn INWARD toward the wall (use inverse growth)
        bl_up_le_l, # BL edge to Upper anchor
        bl_lo_te_l, # BL edge to TE lower
        bl_le_l,    # BL edge to LE point
        bl_wl_ex_l  # Lower wake edge to wake core edge
    ]
    for tag in bl_radial_inward:
        mesh.setTransfiniteCurve(tag, ny_bl, "Progression", bl_growth_ratio_inv)

    # 5b- HORIZONTAL PROGRESSION #
    wake_streamwise_outward = [ # Lines drawn OUTWARD (downstream) from the Trailing Edge
        bl_wt_lo_l, # TE lower to Wake core far
        bl_wl_lo_l  # BL edge lower to Wake lower far
    ]
    for tag in wake_streamwise_outward:
        mesh.setTransfiniteCurve(tag, nx_wake, "Progression", wake_progression)
    wake_streamwise_inward = [ # Lines drawn INWARD (upstream) from the Far Wake toward the Trailing Edge
        bl_wt_up_l, # Wake core far to TE upper
        bl_wu_up_l  # Upper wake far to BL edge upper
    ]
    for tag in wake_streamwise_inward:
        mesh.setTransfiniteCurve(tag, nx_wake, "Progression", wake_progression_inv)

    # 5c- X-AXIS DISCRETIZATION #
    for tag in [le1_curve, bl_le1_l]:     # Upper Leading Edge (Region 1a)
        mesh.setTransfiniteCurve(tag, nx_le1)
    for tag in [le2_curve, bl_le2_l]:     # Lower Leading Edge (Region 1b)
        mesh.setTransfiniteCurve(tag, nx_le2)
    for tag in [upper_curve, bl_up_up_l]: # Upper Airfoil (Region 2a)
        mesh.setTransfiniteCurve(tag, nx_upper, "Bump", chord_bump)
    for tag in [lower_curve, bl_lo_lo_l]: # Lower Airfoil (Region 2b)
        mesh.setTransfiniteCurve(tag, nx_lower, "Bump", chord_bump)

    # 5d- BLUNT TE BASE AND WAKE CORE EXHAUST #
    for tag in [te_blunt_line, bl_wt_ex_l]:
        mesh.setTransfiniteCurve(tag, ny_te)

    # 5e- APPLY TO ALL BOUNDARY LAYER AND WAKE CORE SURFACES #
    bl_surfaces = [
        bl_le1_surface, bl_le2_surface, bl_up_surface, bl_lo_surface, 
        bl_wu_surface, bl_wl_surface, bl_wt_surface
    ]
    for surf in bl_surfaces:
        mesh.setTransfiniteSurface(surf)
        mesh.setRecombine(2, surf)
    

    ## 6- TRANSFINITE MESHING FOR FARFIELD LAYER REGIONS ## 
    # 6a- Farfield Radial Progressions (Vertical/Outward Expansion) #
    ff_radial_outward = [ # Lines drawn OUTWARD from the BL edge
        ff_up_te_l, # BL TE upper to FF TE upper
        ff_lo_le_l, # BL LE lower to FF LE lower
        ff_wu_ex_l  # Upper wake edge to FF exhaust upper
    ]
    for tag in ff_radial_outward:
        mesh.setTransfiniteCurve(tag, ny_farfield, "Progression", ff_growth_ratio)
    ff_radial_inward = [ # Lines drawn INWARD from the FF toward the BL edge
        ff_up_le_l, # FF LE upper to BL LE upper
        ff_lo_te_l, # FF TE lower to BL TE lower
        ff_le_l,    # FF nose to BL nose
        ff_wl_ex_l  # FF exhaust lower to Lower wake edge
    ]
    for tag in ff_radial_inward:
        mesh.setTransfiniteCurve(tag, ny_farfield, "Progression", ff_growth_ratio_inv)

    # 6b- Farfield Streamwise Progressions (Wake Diffusion) #
    # Line drawn OUTWARD (downstream) matching the lower wake
    mesh.setTransfiniteCurve(ff_wl_lo_l, nx_wake, "Progression", wake_progression)
    # Line drawn INWARD (upstream) matching the upper wake
    mesh.setTransfiniteCurve(ff_wu_up_l, nx_wake, "Progression", wake_progression_inv)

    # 6c- Farfield Streamwise Parities (Uniform Airfoil & LE matching) #
    # Upper/Lower Airfoil Parity extended to Farfield (Regions 5a, 5b)
    mesh.setTransfiniteCurve(ff_up_up_l, nx_upper, "Bump", chord_bump)
    mesh.setTransfiniteCurve(ff_lo_lo_l, nx_lower, "Bump", chord_bump)
    # Leading Edge Arc Parity extended to Farfield (Regions 4a, 4b)
    mesh.setTransfiniteCurve(ff_le1_l, nx_le1)
    mesh.setTransfiniteCurve(ff_le2_l, nx_le2)

    # 6d- Apply to Farfield Surfaces #
    ff_surfaces = [
        ff_le1_surface, ff_le2_surface, ff_up_surface, 
        ff_lo_surface, ff_wu_surface, ff_wl_surface
    ]
    for surf in ff_surfaces:
        mesh.setTransfiniteSurface(surf)
        mesh.setRecombine(2, surf)
    #endregion


    #region Outputs and Generation
    ## 7- MARKER DEFINITIONS ##
    # 7a- AIRFOIL SURFACE, NO SLIP WALL #
    airfoil_wall_curves = [lower_curve, le2_curve, le1_curve, upper_curve, te_blunt_line]
    gmsh.model.addPhysicalGroup(1, airfoil_wall_curves, tag=1, name="MARKER_AIRFOIL")

    # 7b- FARFIELD AND EXHAUST BOUNDARIES #
    # This encompasses the entire outer envelope of the mesh
    farfield_curves = [
        ff_up_up_l, ff_lo_lo_l,             # Top and Bottom farfield lines
        ff_le1_l, ff_le2_l,                 # Front leading edge farfield arcs
        ff_wu_ex_l, bl_wt_ex_l, ff_wl_ex_l  # Rear exhaust plane (Upper wake, core wake, lower wake)
    ]
    gmsh.model.addPhysicalGroup(1, farfield_curves, tag=2, name="MARKER_FARFIELD")

    # 7c- FLUID DOMAIN #
    all_fluid_surfaces = bl_surfaces + ff_surfaces
    gmsh.model.addPhysicalGroup(2, all_fluid_surfaces, tag=3, name="FLUID_DOMAIN")

    # 8- SAVE CONFIG AND GENERATE #
    brep_path = os.path.join(working_dir, f"{airfoil.airfoil_name}_geometry.brep")
    gmsh.write(brep_path)
    gmsh.model.mesh.generate(2)
    #endregion

    return brep_path # Out the brep path
