import os
import textwrap

import logging
logger = logging.getLogger(__name__)

def XFoil_Build_Cp(coords_path: str, working_dir: str, alpha: float, Re: float, mach: float, n_panels: int, max_iterations: int) -> tuple[str, str]:
    """
    Builds the XFoil run commands to determine the Cp around the airfoil, returns the path to the input script and Cp output file

    Args:
        coords_path (str): The str path to the geometry file in Selig format, ending with .dat
        working_dir (str): Directory where the XFoil input script and Cp output will be saved
        alpha (float): Angle of attack in degrees
        Re (float): Reynolds number
        mach (float): Mach number
        n_panels (int): Number of panels to be used for the panel method in XFoil
        max_iterations (int): Maximum number of iterations for XFoil to run

    Returns:
        str: Path to the XFoil input script
        str: Path to the Cp output file
    """

    # I/O checks
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        logger.info(f"Created working directory at {working_dir} for XFoil Cp output.")
    cp_file_path = os.path.join(working_dir, "Cp.dat")
    if os.path.exists(cp_file_path):
        logger.warning(f"XFoil Cp output file already exists at {cp_file_path}. It will be overwritten.")
        os.remove(cp_file_path)
    input_script_path = os.path.join(working_dir, "xfoil_cp_input.txt")
    if os.path.exists(input_script_path):
        logger.warning(f"XFoil input script already exists at {input_script_path}. It will be overwritten.")
        os.remove(input_script_path)
    
    commands = textwrap.dedent(f"""
    load {coords_path}
    ppar
    n {n_panels}
    

    oper 
    visc {Re}
    mach {mach}
    iter {max_iterations}
    alfa {alpha}
    cpwr {cp_file_path}

    quit
    """).strip()

    with open(input_script_path, "w") as f:
        f.write(commands)

    return input_script_path, cp_file_path