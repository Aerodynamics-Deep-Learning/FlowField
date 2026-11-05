import os

def GMSH_Write_Exception(exception: Exception, working_dir: str) -> str:
    """
    Logs the entire exception string into a .txt file, returning the path to that file.
    Creates `working_dir` if the orchestrator has not, so the diagnostic always lands somewhere.
    """
    os.makedirs(working_dir, exist_ok=True) # Failsafe for when the dir does not exist

    # Type-qualified: str() alone is empty for a bare raise, e.g. RuntimeError()
    exception_str = f"{type(exception).__name__}: {exception}"
    exception_file_path = os.path.join(working_dir, "gmsh_exception.txt")
    with open(exception_file_path, "w") as f:
        f.write(exception_str)
    return exception_file_path