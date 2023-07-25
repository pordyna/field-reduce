import argparse
import json

from .OutputReducer import OutputReducer

def main():
    # Define command line arguments:
    parser = argparse.ArgumentParser(
        description="Reads an openPMD series and reduces fields(meshes) resolution by pixel binning. The output is "
                    "written into another openPMD series. Attributes are preserved as well. "
                    "There is no support for patches/particles, they will be ignored and will not be saved. "
                    "The main use case is to read from an SST stream series and save to a file based series to reduce "
                    "the amount of data written to disk. Though it should work with other combinations like "
                    "file -> file or stream -> stream as well."
    )
    parser.add_argument("source_path",
                        help="Path to the .sst file of the input stream, or alternatively to a file based openPMD"
                             " series.",
                        type=str)
    parser.add_argument("output_path",
                        help="Path to where the new series should be created. Should include sth like /Data_%%T.bp "
                             "at the end, to specify the backend.",
                        type=str)
    parser.add_argument("-x", "--div-x",
                        help="The number of cells in x directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in x direction in the source. (when calling with mpi "
                             "this has to be true for each chunk, global extend is sliced along the first axis)",
                        type=int,
                        default=1)
    parser.add_argument("-y", "--div-y",
                        help="The number of cells in y directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in y direction in the source. (when calling with mpi "
                             "this has to be true for each chunk, global extend is sliced along the first axis)",
                        type=int,
                        default=1)
    parser.add_argument("-z", "--div-z",
                        help="The number of cells in z directions will be reduced by this value. Has to be an integer."
                             "Has to divide the number of cells in z direction in the source. (when calling with mpi "
                             "this has to be true for each chunk, global extend is sliced along the first axis)",
                        type=int,
                        default=1)
    parser.add_argument('-m', '--meshes', nargs='+', type=str, default=[],
                        help="Meshes should have the reduction applied. Can't be used together with --exclude. "
                             "If not set all, but for ones listed with --exclude, meshes will be processed. "
                             "Note, other meshes will be still copied in their original resolution.")
    parser.add_argument('-e', '--exclude', nargs='+', type=str, default=[],
                        help="A list of meshes to exclude from reduction. Can't be used together with --meshes. "
                             "Note, these meshes will be still copied in their original resolution.")
    parser.add_argument("-w", "--wait", action='store_true',
                        help="When set the script will wait until the source path points to an existing file. "
                             "Use this when the source path points to an .sst file and the writer may not have yet"
                             " created it when the script is trying to open the series.")
    parser.add_argument("-s", "--source-config-path",
                        help="Path to an .json file that specifies the backend specific configuration for the "
                             "source openPMD series.", default='{}',
                        type=str)
    parser.add_argument("-o", "--output-config-path",
                        help="Path to an .json file that specifies the backend specific configuration for the "
                             "output openPMD series.", default='{}',
                        type=str)
    parser.add_argument("--last_iteration", help="Last iteration to process, so that the reader won't wair for new files after this iteration (usefull for ADIOS2 with steps)",
                        default=-1, type=int)  
    parser.add_argument("--first_iteration", help="First iteration to process.",
                        default=-1, type=int)  
    args = parser.parse_args()

    with open(args.source_config_path, 'r') as json_data:
        options_input_string = json.dumps(json.load(json_data))
    with open(args.output_config_path, 'r') as json_data:
        options_output_string = json.dumps(json.load(json_data))
    if args.meshes:
        meshes = args.meshes
    else:
        meshes = None
    if args.exclude:
        exclude = args.exclude
    else:
        exclude = None
    reducer = OutputReducer(args.source_path, args.output_path, args.div_x, args.div_y, args.div_z, meshes, exclude,
                            args.wait,
                            options_input_string, options_output_string, args.last_iteration, args.first_iteration)
    print("Successfully initialized. Input and output series are open. Running now!")
    reducer.run()
    reducer.finalize()
    #del reducer


if __name__ == "__main__":
    main()
