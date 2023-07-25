# field-reduce

```
$ field-reduce --help

usage: field-reduce [-h] [-x DIV_X] [-y DIV_Y] [-z DIV_Z] [-m MESHES [MESHES ...]] [-e EXCLUDE [EXCLUDE ...]] [-w] [-s SOURCE_CONFIG_PATH] [-o OUTPUT_CONFIG_PATH] [--last_iteration LAST_ITERATION] [--first_iteration FIRST_ITERATION] source_path output_path

Reads an openPMD series and reduces fields(meshes) resolution by pixel binning. The output is written into another openPMD series.  Attributes are preserved as well. There is no support for patches/particles, they will be ignored and will not be saved. The main use case is to read from an SST stream series and save to a file based series to reduce the amount of data written to disk. Though it should work with other combinations like file -> file or stream -> stream as well.

positional arguments:
  source_path           Path to the .sst file of the input stream, or alternatively to a file based openPMD series.
  output_path           Path to where the new series should be created. Should include sth like /Data_%T.bp at the end, to specify the backend.

optional arguments:
  -h, --help            show this help message and exit
  -x DIV_X, --div-x DIV_X
                        The number of cells in x directions will be reduced by this value. Has to be an integer.Has to divide the number of cells in x direction in the source. (when calling with mpi this has to be true for each chunk, global extend is sliced along the first axis)
  -y DIV_Y, --div-y DIV_Y
                        The number of cells in y directions will be reduced by this value. Has to be an integer.Has to divide the number of cells in y direction in the source. (when calling with mpi this has to be true for each chunk, global extend is sliced along the first axis)
  -z DIV_Z, --div-z DIV_Z
                        The number of cells in z directions will be reduced by this value. Has to be an integer.Has to divide the number of cells in z direction in the source. (when calling with mpi this has to be true for each chunk, global extend is sliced along the first axis)
  -m MESHES [MESHES ...], --meshes MESHES [MESHES ...]
                        Meshes should have the reduction applied. Can't be used together with --exclude. If not set all, but for ones listed with --exclude, meshes will be processed. Note, other meshes will be still copied in their original resolution.
  -e EXCLUDE [EXCLUDE ...], --exclude EXCLUDE [EXCLUDE ...]
                        A list of meshes to exclude from reduction. Can't be used together with --meshes. Note, these meshes will be still copied in their original resolution.
  -w, --wait            When set the script will wait until the source path points to an existing file. Use this when the source path points to an .sst file and the writer may not have yet created it when the script is trying to open the series.
  -s SOURCE_CONFIG_PATH, --source-config-path SOURCE_CONFIG_PATH
                        Path to an .json file that specifies the backend specific configuration for the source openPMD series.
  -o OUTPUT_CONFIG_PATH, --output-config-path OUTPUT_CONFIG_PATH
                        Path to an .json file that specifies the backend specific configuration for the output openPMD series.
  --last_iteration LAST_ITERATION
                        Last iteration to process, so that the reader won't wair for new files after this iteration (usefull for ADIOS2 with steps)
  --first_iteration FIRST_ITERATION
                        First iteration to process.
```

**Note:** The particle data are ignored and won't be copied to the new output.

**Tipp:** When dealing with  a non streaming input series with many iterations it may ne usefull to disable initial iteration parsing. See `example_configs/in.json`.