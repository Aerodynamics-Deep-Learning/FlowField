from typing import Any

def _write_cfg(cfg_dict: dict[str, dict[str, Any]], cfg_path: str) -> None:
    """
    Writes the config dict given the cfg_dict and cfg_path in the SU2 expected .cfg format

    Args:
        cfg_dict (dict[str, dict[str, Any]]): The config dict to be written to the .cfg file, with keys as sections and values as dicts of key-value pairs for that section
        cfg_path (str): The path to write the .cfg file to

    Returns:
        None, writes the .cfg file to the given path
    """
    with open(cfg_path, 'w') as f:

        for section_idx, (section_name, section_dict) in enumerate(cfg_dict.items()):

            # New line for readibility
            if section_idx > 0:
                f.write("\n")
            
            # Write the section name
            f.write(f"% --- {section_name} ---\n")

            # Unpack the dict and write
            for param, value in section_dict.items():
                f.write(f"{param}= {value}\n")


