import os


def C2D_Write_Exception(exception: Exception, working_dir: str) -> str:
    """
    Logs the entire exception string into a .txt file, returning the path to that file.
    Literal mirror of `gmsh.io.GMSH_Write_Exception`, used by `C2D_MeshGenerator`'s
    blanket `except Exception` the same way GMSH's is used.
    """
    os.makedirs(working_dir, exist_ok=True) # This ensures if things fail catastrophically,
                                            # the diagnostics can at least land somewhere

    # Type-qualified: str() alone is empty for a bare raise, e.g. RuntimeError()
    exception_str = f"{type(exception).__name__}: {exception}"
    exception_file_path = os.path.join(working_dir, "c2d_exception.txt")
    with open(exception_file_path, "w") as f:
        f.write(exception_str)
    return exception_file_path


def write_airfoil_dat(airfoil, path: str) -> str:
    """
    Write an in-memory `Airfoil` (Selig-format coords_tensor, same ordering gmsh consumes) out
    to an XFOIL-format `.dat` file, so c2d's exe (which only reads files) can use it.

    Written unscaled, unlike gmsh's CAD points: the exe re-normalizes whatever it reads back to
    unit chord.

    Returns the path written.
    """
    coords = airfoil.coords_tensor.detach().cpu().numpy()
    with open(path, "w") as f:
        f.write(f"{airfoil.airfoil_name}\n")
        for x, y in coords:
            f.write(f"{float(x): .8f} {float(y): .8f}\n")
    return path
