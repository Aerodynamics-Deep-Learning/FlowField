from typing import Union
import copy

from ....schemas import Airfoil, Freestream
from ..schemas import SU2_SolverConfig
from .utils import _deep_update

def _SU2_BuildCfg_Warm_Euler(base_dict: dict, freestream: Freestream, airfoil: Airfoil, SolverConfig: SU2_SolverConfig) -> dict[str, dict[str, Union[str, int, float]]]:
    """
    Builds the euler warm start config from the base dict and the given airfoil and freestream conditions.

    Args:
        base_dict (dict): The base config dict to be updated for the cold start, unused
        freestream (Freestream): The freestream conditions to be used in the config, unused
        airfoil (Airfoil): The airfoil geometry to be used in the config
        SolverConfig (SU2_SolverConfig): The solver configuration

    Returns:
        dict[str, dict[str, Union[str, int, float]]]: The cold start config dict to be written to the .cfg file
    """

    config = copy.deepcopy(base_dict)

    euler_updates = {
        "PHYSICAL PROBLEM": {
            "SOLVER": "EULER",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
        },
        "BOUNDARY MARKERS": {
            "MARKER_EULER": f"( {SolverConfig.marker_airfoil} )",
        },
        "MULTIGRID PARAMETERS": {
            "MGLEVEL": 3,
        },
        "TIME DISCRETIZATION": {
            "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            "CFL_NUMBER": SolverConfig.cfl_number_euler,
            "CFL_ADAPT": "YES",
            "CFL_ADAPT_PARAM": f"( {SolverConfig.cfl_factordown_euler}, {SolverConfig.cfl_factorup_euler}, {SolverConfig.cfl_min_euler}, {SolverConfig.cfl_max_euler} )",
            "MAX_UPDATE_FLOW": SolverConfig.cfl_maxupdate_euler,
        },
        "SPATIAL DISCRETIZATION": {
            "CONV_NUM_METHOD_FLOW": "JST",
            "JST_SENSOR_COEFF": "( 0.5, 0.02 )",
        },
    }
    
    return _deep_update(config, euler_updates)

def _SU2_BuildCfg_Warm_Restart(base_dict: dict, freestream: Freestream, airfoil: Airfoil, SolverConfig: SU2_SolverConfig) -> dict[str, dict[str, Union[str, int, float]]]:
    """
    Builds the rans warm start config from the base dict and the given airfoil and freestream conditions.

    Args:
        base_dict (dict): The base config dict to be updated for the cold start
        freestream (Freestream): The freestream conditions to be used in the config
        airfoil (Airfoil): The airfoil geometry to be used in the config
        SolverConfig (SU2_SolverConfig): The solver configuration

    Returns:
        dict[str, dict[str, Union[str, int, float]]]: The cold start config dict to be written to the .cfg file
    """

    config = copy.deepcopy(base_dict)

    turb_comp = "YES" if freestream.mach > 0.5 else "NO"

    restart_updates = {
        "INPUT / OUTPUT": {
            "SOLUTION_FILENAME": "flow"
        },
        "PHYSICAL PROBLEM": {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "YES",
        },
        "FREESTREAM PARAMETERS": {
            "REYNOLDS_NUMBER": freestream.Re,
            "REYNOLDS_LENGTH": airfoil.chord,
        },
        "TURBULENCE PHYSICS": {
            "TURB_COMPRESSIBILITY_CORRECTION": turb_comp,
        },
        "BOUNDARY MARKERS": {
            "MARKER_HEATFLUX": f"( {SolverConfig.marker_airfoil}, 0.0 )",
        },
        "MULTIGRID PARAMETERS": {
            "MGLEVEL": 2,
        },
        "TIME DISCRETIZATION": {
            "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
            "TIME_DISCRE_TURB": "EULER_IMPLICIT",
            "CFL_NUMBER": SolverConfig.cfl_number_warm,
            "CFL_ADAPT": "YES",
            "CFL_ADAPT_PARAM": f"( {SolverConfig.cfl_factordown_warm}, {SolverConfig.cfl_factorup_warm}, {SolverConfig.cfl_min_warm}, {SolverConfig.cfl_max_warm} )",
            "CFL_REDUCTION_TURB": SolverConfig.cfl_turbreduction_warm,
            "MAX_UPDATE_FLOW": SolverConfig.cfl_maxupdate_warm,
        },
        "SPATIAL DISCRETIZATION": {
            "CONV_NUM_METHOD_FLOW": "JST",
            "JST_SENSOR_COEFF": "( 0.5, 0.02 )",
            "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND",
            "SPATIAL_ORDER_TURB": "1ST_ORDER",
        },
        "CONVERGENCE PARAMETERS": {
            "CONV_RESIDUAL_MINVAL": SolverConfig.conv_residual_minval_euler,
            "CAUCHY_ELEMS": SolverConfig.cauchy_elems_euler,
            "CAUCHY_EPS": SolverConfig.cauchy_eps_euler,
            "INNER_ITER": SolverConfig.max_iterations_euler,
        },
    }
    
    return  _deep_update(config, restart_updates)
