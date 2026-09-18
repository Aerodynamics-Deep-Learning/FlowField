"""
Headless mesh generation: airfoil .dat -> C2D -> SU2/VTK + quality.

This is the entry point a pipeline calls. It runs the C2D
executable, converts the output to SU2 (and VTK), and returns the
output paths plus a quality summary.
"""

import os
import subprocess

from src.datagen.schemas import Airfoil
from src.datagen.meshing.c2d.convert import convert, su2_to_vtk
from src.datagen.meshing.c2d.io import write_airfoil_dat, C2D_Write_Exception
from src.datagen.meshing.c2d.schemas import C2D_In, C2D_Out, C2D_ExitFlag, C2D_MeshingConfig
from src.datagen.meshing.c2d.utils import C2D_count_su2_nodes

import logging
logger = logging.getLogger(__name__)


def C2D_MeshGenerator(data: C2D_In) -> C2D_Out:
    """
    Orchestrates the entire pipeline of mesh generation using C2D, including flagging,
    io, etc. 
    
    Args:
        data (C2D_In): The input data schema defined for C2D, including the meshing 
            configs, airfoil geometry, freestream, and utils such as io
    
    Returns:
        C2D_Out: The output data schema defined for C2D, including the airfoil, 
            freestream, falg, mesh path, other stuff, and a verbose list
    """
    name = data.airfoil.airfoil_name

    # working_dir is the orchestrator's to mint and create, matching GMSH_MeshGenerator
    # Infer the save paths up front
    log_path = os.path.join(data.working_dir, f"{name}_c2d_log.txt")
    dat_path = os.path.join(data.working_dir, f"{name}.dat") 
    log_text = None  # Fallback value, in case generation failed before any log was captured

    exe = C2D_find_exe()
    if not exe:
        # No exe found, nothing written, simple flag
        return C2D_Out(
            airfoil=data.airfoil, freestream=data.freestream,
            flag=C2D_ExitFlag.EXECUTABLE_NOT_FOUND,
            verbose_list=[None, None, None],
        )

    try:
        # Try getting the mesh
        p3d_path, su2_path, vtk_path, nmf_path, log_text = C2D_generate_mesh(
            airfoil=data.airfoil, exe=exe, working_dir=data.working_dir,
            config=data.meshing_config
        )

        if p3d_path is None:
            return C2D_Out(
                airfoil=data.airfoil, freestream=data.freestream,
                flag=C2D_ExitFlag.SUBPROCESS_FAIL,
                verbose_list=[log_path, dat_path, None],
            )

        if su2_path is None:
            return C2D_Out(
                airfoil=data.airfoil, freestream=data.freestream,
                flag=C2D_ExitFlag.CONVERSION_FAIL,
                verbose_list=[log_path, dat_path, p3d_path],
            )

        # Structural check only, mirroring GMSH's num_quads==0: zero nodes means conversion
        # nominally succeeded but produced no usable mesh. Quality is scored centrally by
        # common.quality.Common_evaluate_mesh_quality from the written .su2, like in GMSH.
        num_nodes = C2D_count_su2_nodes(su2_path)
        flag = C2D_ExitFlag.CONVERSION_FAIL if num_nodes == 0 else C2D_ExitFlag.SUCCESS

        return C2D_Out(
            airfoil=data.airfoil, freestream=data.freestream, flag=flag,
            mesh_path=su2_path, mesh_path_vtk=vtk_path, num_nodes=num_nodes,
            verbose_list=[log_path, dat_path, nmf_path],
        )

    except Exception as e:
        # Build the verbose list and exception. The writer creates the dir it needs, but can still
        # fail on permissions or an unusable path, it must not replace the real exception with an IO one.
        try:
            exception_path = C2D_Write_Exception(e, data.working_dir)
        except Exception:
            logger.exception("Could not write the c2d exception file to %s", data.working_dir)
            exception_path = None
        verbose_list = [log_path, dat_path, exception_path]

        return C2D_Out(
            airfoil=data.airfoil, freestream=data.freestream,
            flag=C2D_ExitFlag.FATAL_ERROR,
            verbose_list=verbose_list
        )

    finally:
        try:
            with open(log_path, "w") as log_file:
                log_file.write(log_text or "")
        except Exception:
            pass


def _te_is_sharp(dat_path: str, tol: float = 1e-5) -> bool:
    """True if the airfoil's first and last points coincide (sharp TE)."""
    pts = []
    with open(dat_path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2:
                try:
                    pts.append((float(t[0]), float(t[1])))
                except ValueError:
                    pass
    if len(pts) < 3:
        return False
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5 < tol


def _write_grid_options(path: str, params: dict, name: str, topo: str, slvr: str) -> None:
    """Writes C2D's `grid_options.in` namelist directly from `params` (expected to
    already be complete, `C2D_MeshingConfig.to_params()` always supplies every field)."""
    fmt = lambda v: str(int(v)) if isinstance(v, int) else repr(float(v))
    with open(path, "w") as f:
        f.write("&SOPT\n")
        for k in ("nsrf", "lesp", "tesp", "radi", "nwke", "fdst", "fwkl", "fwki"):
            f.write("  %s = %s\n" % (k, fmt(params[k])))
        f.write("/\n&VOPT\n  name = '%s'\n" % name)
        for k in ("jmax", "ypls", "recd", "stp1", "stp2", "nrmt", "nrmb",
                  "alfa", "epsi", "epse", "funi", "asmt", "cfrc"):
            f.write("  %s = %s\n" % (k, fmt(params[k])))
        f.write("  slvr = '%s'\n  topo = '%s'\n" % (slvr, topo))
        f.write("/\n&OOPT\n  gdim = 2\n  npln = 2\n  dpln = 1.0\n/\n")


def C2D_generate_mesh(airfoil: Airfoil, exe: str, working_dir: str, config: C2D_MeshingConfig,
                      timeout: int = 300):
    """
    Writes the airfoil to a `.dat` file, drives the C2D binary directly, and converts the
    resulting grid to SU2/VTK. Mirrors `gmsh.run.GMSH_generate_cmesh`/`gmsh.run.GMSH_generate_omesh`'s roles 
    in `GMSH_MeshGenerator`: performs the actual meshing work and returns only the artifact paths the
    caller needs, flagging and exception handling stay in `C2D_MeshGenerator`, the same split
    GMSH uses.

    Returns (p3d_path, su2_path, vtk_path, nmf_path, log_text). `su2_path`/`vtk_path`/`nmf_path`
    are None if conversion didn't yield a `.su2`; all four paths are None if C2D itself
    never produced a `.p3d`. `log_text` is C2D's captured stdout/stderr (or the caught
    launch/timeout error) and is always populated, mirroring how GMSH's own logger output is
    always available for `GMSH_MeshGenerator` to dump regardless of outcome.
    """
    name = airfoil.airfoil_name
    dat_path = os.path.join(working_dir, f"{name}.dat")
    write_airfoil_dat(airfoil, dat_path)

    # c2d prompts for an interactive y/n confirmation when the topo in grid_options.in
    # disagrees with what it recommends for this airfoil's TE shape ("Sharp trailing edge: C-grid
    # topology is recommended." / "Blunt trailing edge: O-grid topology is recommended.").
    # Always honor the caller's requested topo as-is, warn that the mismatch may hurt mesh quality, 
    # and auto-answer c2d's confirmation prompt so this headless pipeline doesn't hang waiting on it.
    topo = config.topo; slvr = config.slvr
    recommended = "CGRD" if _te_is_sharp(dat_path) else "OGRD"
    confirm = ""
    if topo != recommended: # If the current setup does not match expected recommendation, do what needs to be done
        logger.warning(
            "%s: requested topo=%s but TE shape recommends topo=%s; auto-confirming c2d's "
            "override prompt (mesh quality may suffer from the mismatch)",
            name, topo, recommended,
        )
        confirm = "y\n"

    _write_grid_options(os.path.join(working_dir, "grid_options.in"), config.to_params(),
                        name, topo, slvr)
    stdin_cmds = os.path.basename(dat_path) + "\n" + confirm + "grid\nsmth\nquit\n"

    try:
        proc = subprocess.run([exe], cwd=working_dir, input=stdin_cmds, text=True,
                              timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired as e:
        log_text = ("c2d timed out after %ss\n--- stdout ---\n%s\n--- stderr ---\n%s"
                    % (timeout, e.stdout or "", e.stderr or ""))
        return None, None, None, None, log_text
    except (OSError, subprocess.SubprocessError) as e:
        return None, None, None, None, "c2d failed to launch: %s: %s" % (type(e).__name__, e)

    log_text = ("c2d exited %s\n--- stdout ---\n%s\n--- stderr ---\n%s"
               % (proc.returncode, proc.stdout, proc.stderr))
    p3d_path = os.path.join(working_dir, name + ".p3d")
    if not os.path.isfile(p3d_path):
        return None, None, None, None, log_text

    base = os.path.join(working_dir, name)
    nmf_path, su2_path, vtk_path = base + ".nmf", base + ".su2", base + ".vtk"
    convert(p3d_path, nmf_path, su2_path, verbose=False)
    if not os.path.isfile(su2_path):
        return p3d_path, None, None, None, log_text

    su2_to_vtk(su2_path, vtk_path)
    return p3d_path, su2_path, vtk_path, nmf_path, log_text


def C2D_find_exe(search_dirs=None):
    """Locate the c2d executable (bin/ next to the package by default)."""
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = search_dirs or [os.path.join(here, "bin"), here]
    for d in dirs:
        for name in ("c2d.exe", "c2d"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return os.path.abspath(p)
    return None


