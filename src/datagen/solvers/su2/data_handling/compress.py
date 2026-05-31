"""
Code to handle the reduction of the data footprint, post proof that it converged
"""
from pathlib import Path
from typing import Union

import polars as pl
import lzma

from ..schemas import SU2_SolutionStrategy
from .maps import COLD_MAP, WARM_MAP, MACHSEQ_MAP

def SU2_CompressData(strategy: SU2_SolutionStrategy, working_dir: Union[str, Path], clean_paraview: bool, filename_dict: dict=None) -> None:
    """
    The overall handler for all the data compression procedure. This function compresses .dat/.vtu to .xz, and .csv to .parquet

    Args:
        strategy (SU2_SolutionStrategy): The strategy that was used for the run, to determine the files that need to be compressed
        working_dir (str | Path): The directory where the data is stored, and where the compressed data will be stored
        clean_paraview (bool): Cleans the already decided upon files, goes for a leaner storage if chosen
        filename_dict (dict): A dictionary containing the names of the files to be compressed, with keys 'dat_files', 'vtu_files', and 'csv_files' with a default dict implemented
    
    Returns:
        None, the function will save the compressed files in the same directory as the original files with the same name but with .xz extension for .dat/.vtu files and .parquet extension for .csv files
    """
    working_dir = Path(working_dir)

    if strategy == SU2_SolutionStrategy.COLD:
        # Tree if the strategy is cold start
        # Apply basemap to filename dict if not provided, else check if the provided filename dict has the necessary keys
        if filename_dict is None:
            filename_dict = COLD_MAP
        else:
            if not set(COLD_MAP.keys()).issubset(filename_dict.keys()):
                raise ValueError(f"filename_dict is missing required keys for COLD start. Required: {list(COLD_MAP.keys())}")

        # Convert the csvs to parquets
        # Surface flow
        surface_flow_path = working_dir / filename_dict['surface_flow']
        surface_flow_parquet_path = surface_flow_path.with_suffix('.parquet')
        _convert_to_parquet(surface_flow_path, surface_flow_parquet_path)
        surface_flow_path.unlink(missing_ok=True) # Deletes the big file

        # Entire flow
        flow_csv_path = working_dir / filename_dict['flow_csv']
        flow_csv_parquet_path = flow_csv_path.with_suffix('.parquet')
        _convert_to_parquet(flow_csv_path, flow_csv_parquet_path)
        flow_csv_path.unlink(missing_ok=True) # Deletes the big file

        # Compress the dat and vtu files to xz
        # Restart file, .dat of entire flow
        restart_flow_path = working_dir / filename_dict['restart_flow']
        restart_flow_compressed_path = working_dir / f"{restart_flow_path.name}.xz"
        _compress_to_xz(restart_flow_path, restart_flow_compressed_path)
        restart_flow_path.unlink(missing_ok=True) # Deletes the uncompressed file

        # Visualizable .vtu of entire flow
        flow_vtu_path = working_dir / filename_dict['flow_vtu']
        if clean_paraview:
            flow_vtu_path.unlink(missing_ok=True) # Deletes the uncompressed file, not keeping it at all
        else:
            flow_vtu_compressed_path = working_dir / f"{flow_vtu_path.name}.xz"
            _compress_to_xz(flow_vtu_path, flow_vtu_compressed_path)
            flow_vtu_path.unlink(missing_ok=True) # Deletes the uncompressed file, keeping compressed
            
    elif strategy == SU2_SolutionStrategy.WARM_EULER:
        pass # not implemented yet

    elif strategy == SU2_SolutionStrategy.MACH_SEQ:
        pass # not implemented yet

    else: 
        raise NotImplementedError(f"Data compression not implemented yet for strategy {strategy}")

def _convert_to_parquet(csv_path: Union[str, Path], parquet_path: Union[str, Path]) -> None:
    """
    Takes in flow field data in the .csv format, converts them to .parquet format

    Args:
        csv_path (str | Path): The path to the .csv file to be converted, the .parquet file will be saved in the same directory with the same name but with .parquet extension
        parquet_path (str | Path): The path to save the .parquet file to, if None, will save in the same directory with the same name but with .parquet extension

    Returns:
        None, saves the .parquet file in the same directory as the .csv file
    """
    # Convert to .parquet
    lazy_df = pl.scan_csv(csv_path)
    lazy_df.sink_parquet(parquet_path, compression='zstd')

def _compress_to_xz(file_path: Union[str, Path], compressed_path: Union[str, Path], preset: int=6) -> None:
    # Needs to be swapped out of the linear compute path to ProcessPoolExecutor later on
    """
    Takes in files of format .dat and .vtu, compresses them to .xz using LZMA2 compression

    Args:
        file_path (str | Path): The path to the file to be compressed, the compressed file will be saved in the same directory with the same name but with .xz extension
        compressed_path (str | Path): The path to save the compressed file to, if None, will save in the same directory with the same name but with .xz extension
        preset (int): Compute to compress ratio, 6 is mid, 9 is heavily compression agressive

    Returns:
        None, saves the compressed file in the same directory as the original file
    """
    with open(file_path, 'rb') as f_in:
        data = f_in.read()
    
    compressed_data = lzma.compress(data, format=lzma.FORMAT_XZ , preset=preset)

    with open(compressed_path, 'wb') as f_out:
        f_out.write(compressed_data)
