import os

def GMSH_Write_Exception(exception: Exception, working_dir: str) -> str:
    """
    Logs the entire exception string into a .txt file, returning the path to that file
    """

    exception_str = str(exception)
    exception_file_path = os.path.join(working_dir, "gmsh_exception.txt")
    with open(exception_file_path, "w") as f:
        f.write(exception_str)
    return exception_file_path