"""
Purpose: To make sure for polar plots, we have regression parity
 
Combines a ground truth polar plot, and a produced polar plot (via running specified sweeps) to compare the two

prevents: major regression parity problems, i.e. wrongly applied physics/solver
exists for: running for the full extent, plotting the polar(s) for both gt, produced, and their difference
"""
