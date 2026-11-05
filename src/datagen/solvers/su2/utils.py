from .schemas import SU2_SolutionStrategy
from ...schemas import Freestream

def SU2_StrategyIdentify(Freestream: Freestream):
    """
    Handler to identify the solver strategy to build configs built and run, given the freestream vals

    Args:
        Freestream (Freestream): The freestream config vals, picks AoA, Re, Mach from this

    Returns:
        SU2_SolutionStrategy: The flag of strategy for the specific route chosen
    """
    if abs(Freestream.alpha) >= 10.0 or (abs(Freestream.alpha) >= 8.0 and Freestream.mach >= 0.6):
        # AoA >= 10 OR (AoA >= 8.0 AND M >= 0.6)
        return SU2_SolutionStrategy.MACH_SEQ
    
    elif Freestream.mach >= 0.55 and Freestream.Re >= 3.0e6:
        # M >= 0.55 AND Re >= 3.0e6 AND NOT MACH_SEQ
        return SU2_SolutionStrategy.WARM_EULER
    
    else:
        # NOT MACH_SEQ AND NOT WARM_EULER
        return SU2_SolutionStrategy.COLD # default to cold start
    

def SU2_KeepDeleteVTU() -> bool:
    """
    Decides on either keeping paraview data to be able to conduct visualization or delete it, false for now until implementation
    """
    return False
