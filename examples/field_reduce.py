import sys
import argparse
import openpmd_api as io

from ..OutputReducer import OutputReducer


def main():
    parser = argparse.ArgumentParser(
        description="Reads an openPMD SST stream and reduced fields(meshes) resolution by pixel binning and moves "
                    "attributes and reduced meshes (no support for patches/particles) "
                    "into a file based openPMD series.")
    parser.add_argument("--sst_file",
                        help="Path to the .sst file of the input stream",
                        type=str)
    parser.add_argument("--output_path",
                        help="Path to where the new series should be created. Should include sth like /Data_%T.bp "
                             "at the end, to specify the backend.",
                        type=str)
    parser.add_argument("--div_x",
                        help="The number of cells in x directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in x direction in the source.",
                        type=int,
                        default=1)
    parser.add_argument("--div_y",
                        help="The number of cells in y directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in y direction in the source.",
                        type=int,
                        default=1)
    parser.add_argument("--div_z",
                        help="The number of cells in z directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in z direction in the source.",
                        type=int,
                        default=1)
    parser.add_argument("--div_z",
                        help="The number of cells in z directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in z direction in the source.",
                        type=int,
                        default=1)
    parser.add_argument('-m', '--meshes', nargs='+', default=[])
    args = parser.parse_args()

if __name__ == "__main__":
    if 'adios2' not in io.variants or not io.variants['adios2']:
        print('This requires ADIOS2')
        sys.exit(0)

    backends = io.file_extensions
    if "sst" not in backends:
        print("SST engine not available in ADIOS2.")
        sys.exit(0)

    OutputReducer()
