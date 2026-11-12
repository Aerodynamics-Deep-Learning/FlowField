"""
The design vector: the 18 free CST values a design is sampled, stored and learned as.

Upper A_0..A_8, lower A_1..A_8 (its A_0 is -upper A_0, `CSTParams`' shared nose) and the LE weight.
"""

import numpy as np

from src.datagen.geometry.common.constants import CST_N1, CST_N2, CST_ORDER
from src.datagen.geometry.common.schemas import CSTParams, Nose


# Length of the model's input: 9 upper + 8 lower + the LE weight at order 8
N_FREE = 2 * CST_ORDER + 2

def AL_cst_to_free(cst: CSTParams) -> np.ndarray:
    """
    The 18 free values of a design: upper A_0..A_N, lower A_1..A_N, the LE weight.

    Args:
        cst (CSTParams): A LEM shape at CST_ORDER with the pipeline's class exponents

    Returns:
        np.ndarray: (N_FREE,) float64

    Raises:
        ValueError: For another nose (SEPARATE has a second, independent nose), order, or n1/n2,
            which the vector leaves out: two shapes differing only there would share one
    """
    if cst.nose is not Nose.LEM:
        raise ValueError(f"designs are LEM, got nose={cst.nose.value}: a SEPARATE shape has two noses, "
                         f"not the {N_FREE} free values p_mesh takes")
    if cst.order != CST_ORDER:
        raise ValueError(f"designs are order {CST_ORDER}, got {cst.order}")
    if (cst.n1, cst.n2) != (CST_N1, CST_N2):
        raise ValueError(f"designs have class exponents n1={CST_N1}, n2={CST_N2}, got n1={cst.n1}, n2={cst.n2}: "
                         f"the free values leave them out")
    return np.array([*cst.upper, *cst.lower[1:], cst.le_weight], dtype=np.float64)


def AL_free_to_cst(x: np.ndarray) -> CSTParams:
    """
    The design behind 18 free values, `AL_cst_to_free`'s inverse; `CSTParams` derives the lower A_0.

    Args:
        x (np.ndarray): (N_FREE,) free values

    Returns:
        CSTParams: The LEM shape
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (N_FREE,):
        raise ValueError(f"need {N_FREE} free values, got shape {x.shape}")
    n = CST_ORDER + 1
    return CSTParams(n1=CST_N1, n2=CST_N2, upper=x[:n].tolist(), lower=x[n:2 * n - 1].tolist(),
                     le_weight=float(x[-1]), nose=Nose.LEM)