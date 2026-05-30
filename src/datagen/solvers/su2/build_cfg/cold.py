from typing import Union

from ....schemas import Airfoil, Freestream
from .utils import _deep_update

def _SU2_BuildCfg_Cold(base_dict: dict, freestream: Freestream, airfoil: Airfoil) -> dict[str, dict[str, Union[str, int, float]]]:
    """
    Builds the cold start config from the base dict and the given airfoil and freestream conditions.

    Args:
        base_dict (dict): The base config dict to be updated for the cold start
        freestream (Freestream): The freestream conditions to be used in the config
        airfoil (Airfoil): The airfoil geometry to be used in the config

    Returns:
        dict[str, dict[str, Union[str, int, float]]]: The cold start config dict to be written to the .cfg file
    """

    # Get if stated stuff
    numerics = _get_numerics(freestream)
    turb_comp = "YES" if freestream.mach > 0.5 else "NO"

    cold_updates = {
        "PHYSICAL PROBLEM": {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
        },
        "FREESTREAM PARAMETERS": {
            "REYNOLDS_NUMBER": freestream.Re,
            "REYNOLDS_LENGTH": airfoil.chord,
        },
        "TURBULENCE PHYSICS": {
            "TURB_COMPRESSIBILITY_CORRECTION": turb_comp,
        },
        "TIME DISCRETIZATION": {
            "TIME_DISCRE_TURB": "EULER_IMPLICIT",
        },
        "SPATIAL DISCRETIZATION": {
            **numerics,
            "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND"
        },
    }

    return _deep_update(base_dict, cold_updates)

def _get_numerics(freestream: Freestream) -> dict:
    """
    Gets the numerics for the cold start config based on the freestream conditions, as recommended by SU2 docs

    Args:
        freestream (Freestream): The freestream conditions to be used in the config, picks M from this
    
    Returns:
        dict: The numerics dict to be used in the config, with keys "CONV_NUM_METHOD_FLOW" and either "ENTROPY_FIX_COEFF" or "JST_SENSOR_COEFF"
    """
    if freestream.mach > 0.7:
        numerics = {
            "CONV_NUM_METHOD_FLOW": "ROE",
            "ENTROPY_FIX_COEFF": 0.1,
        }
    else:
        numerics = {
            "CONV_NUM_METHOD_FLOW": "JST",
            "JST_SENSOR_COEFF": "( 0.5, 0.02 )",
        }
    return numerics
