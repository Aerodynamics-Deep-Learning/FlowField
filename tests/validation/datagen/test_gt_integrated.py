"""
Purpose: To make sure (for a single run case) that we have regression parity
 
Combines ground truth solution, and the produced solution (via airfoil, freestream, etc.) to compare the two

prevents: major problems caused by eiter the meshing or solving confgis that prevent regression parity
exists for: running for the full extent, plotting the flow field(s) for both gt, produced, and their difference
"""
