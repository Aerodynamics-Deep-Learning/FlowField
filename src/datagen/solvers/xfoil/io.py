import torch

def XFoil_Parse_Coords(coords_path: str) -> torch.Tensor:
    """
    Convers the raw .dat file of the coords into a tensor of the form [N, [x, y]] in the Selig format

    Args:
        coords_path (str): The path string to the coords

    Returns:
        torch.Tensor: The tensor of the form [N, [x, y]] in the Selig format, representing the coords of the airfoil
    """
    coords_temp = []
    with open(coords_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x, y = map(float, parts[:2])
                    coords_temp.append([x, y])
                except ValueError:
                    continue
    return torch.tensor(coords_temp, dtype=torch.float32)

def XFoil_Parse_Cp(cp_file_path: str) -> torch.Tensor:
    """
    Parses the outputted Cp file from XFoil, t get the final Cp distribution tensor of the form [N, [x, Cp]] in Selig

    Args:
        cp_file_path (str): The path string to the Cp file outputted by XFoil

    Returns:
        torch.Tensor: The tensor of the form [N, [x, Cp]] in the Selig format, representing the Cp distribution around the airfoil
    """
    Cp_temp = []
    with open(cp_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    Cp = float(parts[2])
                    Cp_temp.append([x, Cp])
                except ValueError:
                    continue
    return torch.tensor(Cp_temp, dtype=torch.float32)
    
def XFoil_Write_Stdout(working_dir: str, stdout: str) -> str:
    """
    Writes the stdout of the entire run, returns the path it was written to

    Args:
        working_dir (str): The path string of the working directory
        stdout (str): The entire stdout printout
    
    Returns:
        stdout_path (str): The path string of the saved stdout
    """
    stdout_file_path = f"{working_dir}/stdout.txt"

    with open(stdout_file_path, "w") as f:
                f.write(stdout)

    return stdout_file_path
