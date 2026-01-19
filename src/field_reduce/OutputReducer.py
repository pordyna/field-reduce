import openpmd_api as api
import os.path
import time
import numpy as np

from typing import Optional, Iterable, Tuple

try:
    from mpi4py import MPI

    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

from .downscale_kernel import downscale


# copied from openpmd-pipe

class FallbackMPICommunicator:
    def __init__(self):
        self.size = 1
        self.rank = 0


class Chunk:
    """
    A Chunk is an n-dimensional hypercube, defined by an offset and an extent.
    Offset and extent must be of the same dimensionality (Chunk.__len__).
    """

    def __init__(self, offset, extent):
        assert (len(offset) == len(extent))
        self.offset = offset
        self.extent = extent

    def __len__(self):
        return len(self.offset)

    def slice1D(self, mpi_rank, mpi_size, divisible=1, dimension=None):
        """
        Slice this chunk into mpi_size hypercubes along one of its
        n dimensions. The dimension is given through the 'dimension'
        parameter. If None, the dimension with the largest extent on
        this hypercube is automatically picked.
        Returns the mpi_rank'th of the sliced chunks.
        """
        if dimension is None:
            # pick that dimension which has the highest count of items
            dimension = 0
            maximum = self.extent[0]
            for k, v in enumerate(self.extent):
                if v > maximum:
                    dimension = k
        assert (dimension < len(self))
        # no offset
        assert (self.offset == [0 for _ in range(len(self))])
        offset = [0 for _ in range(len(self))]
        stride = self.extent[dimension] // mpi_size
        rest = self.extent[dimension] % mpi_size

        # local function f computes the offset of a rank
        # for more equal balancing, we want the start index
        # at the upper gaussian bracket of (N/n*rank)
        # where N the size of the dataset in dimension dim
        # and n the MPI size
        # for avoiding integer overflow, this is the same as:
        # (N div n)*rank + round((N%n)/n*rank)
        def f(rank):
            res = stride * rank
            padDivident = rest * rank
            pad = padDivident // mpi_size
            if pad * mpi_size < padDivident:
                pad += 1
            offset_l = res + pad
            if offset_l % divisible != 0:
                offset_l += divisible - offset_l % divisible
            return offset_l

        offset[dimension] = f(mpi_rank)
        extent = self.extent.copy()
        if mpi_rank >= mpi_size - 1:
            extent[dimension] -= offset[dimension]
        else:
            extent[dimension] = f(mpi_rank + 1) - offset[dimension]

        return Chunk(offset, extent)


def copy_attributes(source: api.Attributable,
                    target: api.Attributable) -> None:
    """ Copies all attributes from one attributable to another one

        :param source: An openPMD-api attributable from that all attributes should be copied
        :param target: An openPMD-api attributable that should receive the attributes
        """
    # ignoring attributes  like in the openpmd-pipe implementation:
    ignored_attributes = {
        api.Series:
            ["basePath", "iterationEncoding", "iterationFormat", "openPMD"],
        api.Iteration: ["snapshot"]
    }
    for attribute in source.attributes:
        ignore_this_attribute = False
        for openpmd_group, to_ignore_list in ignored_attributes.items():
            if isinstance(source, openpmd_group):
                for to_ignore in to_ignore_list:
                    if attribute == to_ignore:
                        ignore_this_attribute = True

        # actual copy
        dtypes = source.attribute_dtypes
        if not ignore_this_attribute:
            target.set_attribute(attribute, source.get_attribute(attribute), dtypes[attribute])


class OutputReducer:

    def __init__(self, source_path: str, output_path: str, div_x: int,
                 div_y: int, div_z: int,
                 meshes: Optional[Iterable[str]] = None,
                 exclude: Optional[Iterable[str]] = None,
                 wait: bool = False, options_in='{}', options_out='{}', last_iteration=-1, first_iteration=-1,
                 checkpoint_path: Optional[str] = None):
        """ Output Reducer initializer

        :param source_path: path to the source openPMD series
        :param output_path: path to the destination openPMD series
        :param div_x: pixels in a bin along x direction
        :param div_y: pixels in a bin along y direction
        :param div_z: pixels in a bin along z direction
        :param meshes: meshes to reduce, other meshes will be simply copied
        :param exclude: meshes to exclude from reduction (alternative to the meshes option)
        :param wait: if true the program will wait until the source path exist (useful for sst set-ups)
        :param options_in: json string with backend specific configuration for the input series
        :param options_out: json string with backend specific configuration for the output series
        :param last_iteration: Last iteration to process, useful for avoiding waiting indefinitely fro a next iteration
            when using adios steps. Set to sth < 0 to disable this check
        :param first_iteration: First iteration to process.
        :param checkpoint_path: Path to checkpoint file to track last successfully processed iteration.
            If None, uses default: field_reduce_checkpoint_<input_filename>
        """
        if meshes is not None and exclude is not None:
            raise ValueError("meshes and exclude are exclusive optional arguments and can't be set at the same time")
        if meshes is not None:
            self.meshes_to_reduce = meshes
        else:
            self.meshes_to_reduce = []
        if exclude is not None:
            self.meshes_to_exclude = exclude
        else:
            self.meshes_to_exclude = []
        self.axis_scaling = {'x': div_x, 'y': div_y, 'z': div_z}
        
        # Setup checkpoint path
        if checkpoint_path is None:
            # Extract filename from source_path (remove any directory path and extension)
            import pathlib
            source_filename = pathlib.Path(source_path).stem
            checkpoint_path = f"field_reduce_checkpoint_{source_filename}"
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint if it exists and first_iteration is at default
        if first_iteration == -1 and os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    last_successful_iteration = int(f.read().strip())
                    first_iteration = last_successful_iteration + 1
                    print(f"Loaded checkpoint: resuming from iteration {first_iteration}", flush=True)
            except (ValueError, IOError) as e:
                print(f"Warning: Failed to read checkpoint file {self.checkpoint_path}: {e}", flush=True)
        
        if HAVE_MPI:
            self.comm = MPI.COMM_WORLD
            print(f"You are using MPI. Welcome from rank {self.comm.rank} of {self.comm.size}", flush=True)
        else:
            self.comm = FallbackMPICommunicator()
            print("In serial mode", flush=True)
        if wait:
            while not os.path.exists(source_path):
                time.sleep(1)
        if self.comm.size == 1:
            self.output_series = api.Series(output_path, api.Access.create, options_out)
        else:
            self.output_series = api.Series(output_path, api.Access.create, self.comm, options_out)
        print("opened output series", flush=True)
        if self.comm.size == 1:
            self.input_series = api.Series(source_path, api.Access.read_only, options_in)
        else:
            self.input_series = api.Series(source_path, api.Access.read_only, self.comm, options_in)
        print("opened input series", flush=True)
        self.stored_meshes: dict[str, dict[str, Tuple[np.ndarray, Tuple]]] = {}
        self.last_iteration = last_iteration
        self.first_iteration = first_iteration

    def finalize(self):
        del self.output_series
        del self.input_series

    def _to_be_reduced(self, mesh_name: str) -> bool:
        """ Checks if a certain mesh is supposed to be reduced

        :param mesh_name: the name of the mesh to check
        :return: true if the mesh should be binned
        """
        # no options set:
        if not self.meshes_to_reduce and not self.meshes_to_exclude:
            return True
        # meshes used
        elif self.meshes_to_reduce:
            return mesh_name in self.meshes_to_reduce
        # exclude used
        else:
            return mesh_name not in self.meshes_to_exclude

    def _process_mrc_before_close(self, input_mrc: api.Mesh_Record_Component,
                                  output_mrc: api.Mesh_Record_Component, mesh_dict: dict, mrc_name: str,
                                  divisible: int) -> None:
        copy_attributes(input_mrc, output_mrc)
        shape = input_mrc.shape
        offset = [0 for _ in shape]
        # determine chunk to process on this mpi rank
        chunk = Chunk(offset, shape)
        local_chunk = chunk.slice1D(self.comm.rank, self.comm.size, dimension=0, divisible=divisible)
        input_data = input_mrc.load_chunk(local_chunk.offset, local_chunk.extent)
        # This should be a view without calling .view() explicitly but better be sure since flushing is done later
        mesh_dict[mrc_name] = (input_data.view(), shape)

    def _process_mesh_before_close(self, input_mesh: api.Mesh, output_mesh: api.Mesh, mesh_dict: dict,
                                   divisible: bool = False) -> None:
        copy_attributes(input_mesh, output_mesh)
        # go over record components:
        if divisible:
            divisible = self.axis_scaling[input_mesh.axis_labels[0]]
        else:
            divisible = 1
        for mrc_name in input_mesh:
            self._process_mrc_before_close(input_mesh[mrc_name], output_mesh[mrc_name], mesh_dict, mrc_name, divisible)

    def run(self):
        copy_attributes(self.input_series, self.output_series)
        write_iterations = self.output_series.write_iterations()
        # process iterations in the input series
        for input_iteration in self.input_series.read_iterations():
            start_time = time.time()
            idx = input_iteration.iteration_index
            if idx < self.first_iteration > 0:
                input_iteration.close()
                print(f"[rank: {self.comm.rank}]:  Skipping iteration number {idx}.",
                      flush=True)
                continue
            print(f"[rank: {self.comm.rank}]:  Starting to process iteration number {idx}."
                  f" Starting to read data from source.", flush=True)
            # create iteration and copy attributes
            output_iteration = write_iterations[idx]
            # copy iteration attributes
            copy_attributes(input_iteration, output_iteration)

            # handle meshes
            input_meshes = input_iteration.meshes
            output_meshes = output_iteration.meshes
            copy_attributes(input_meshes, output_meshes)

            for mesh_name in input_meshes:
                to_be_reduced = self._to_be_reduced(mesh_name)
                mesh_dict = self.stored_meshes[mesh_name] = {}
                self._process_mesh_before_close(input_meshes[mesh_name], output_meshes[mesh_name], mesh_dict,
                                                divisible=to_be_reduced)
                # copy mesh data to memory
                self.input_series.flush()
                mesh = output_iteration.meshes[mesh_name]
                new_grid_spacing = mesh.grid_spacing
                if to_be_reduced:
                    for dd in range(len(new_grid_spacing)):
                        new_grid_spacing[dd] *= self.axis_scaling[mesh.axis_labels[dd]]
                    mesh.set_grid_spacing(new_grid_spacing)
                # the new grid will have larger cells, so we need to adjust grid spacing if the mesh is beeing reduced
                for mrc_name, (mrc_data_old, old_global_shape) in self.stored_meshes[mesh_name].items():
                    if to_be_reduced:
                        offset = [0 for _ in old_global_shape]
                        old_global_chunk = Chunk(offset, old_global_shape)
                        old_local_chunk = old_global_chunk.slice1D(self.comm.rank, self.comm.size,
                                                                   dimension=0,
                                                                   divisible=self.axis_scaling[mesh.axis_labels[0]])
                        local_chunk = old_local_chunk
                        global_chunk = old_global_chunk
                        for dd in range(len(old_global_shape)):
                            local_chunk.extent[dd] = local_chunk.extent[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                            local_chunk.offset[dd] = local_chunk.offset[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                            global_chunk.extent[dd] = global_chunk.extent[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                            global_chunk.offset[dd] = global_chunk.offset[dd] // self.axis_scaling[mesh.axis_labels[dd]]

                        mrc_data = np.empty(shape=local_chunk.extent, dtype=mrc_data_old.dtype)
                        downscale(mrc_data_old, mrc_data)
                    else:
                        offset = [0 for _ in old_global_shape]
                        global_chunk = Chunk(offset, old_global_shape)
                        local_chunk = global_chunk.slice1D(self.comm.rank, self.comm.size, dimension=0, divisible=1)
                        mrc_data = mrc_data_old
                    mrc = mesh[mrc_name]
                    dataset = api.Dataset(mrc_data.dtype, global_chunk.extent)
                    mrc.reset_dataset(dataset)
                    # print(f"[rank: {self.comm.rank}, mesh: {mesh_name}, mrc: {mrc_name}]: "
                    #       f"mrc_data.shape: {mrc_data.shape}, local_chunk.offset: {local_chunk.offset}"
                    #       f" local_chunk.extent {local_chunk.extent}")
                    mrc.store_chunk(mrc_data, local_chunk.offset, local_chunk.extent)
                self.stored_meshes = {}
            input_iteration.close()
            output_iteration.close()
            elapsed = time.time() - start_time
            print(
                f"[rank: {self.comm.rank}]: Finished processing iteration number {idx}. Took {elapsed // 60} m {elapsed % 60} s",
                flush=True)
            # Write checkpoint after successful iteration
            if self.comm.rank == 0:
                try:
                    with open(self.checkpoint_path, 'w') as f:
                        f.write(str(idx))
                except IOError as e:
                    print(f"Warning: Failed to write checkpoint file {self.checkpoint_path}: {e}", flush=True)
            if idx >= self.last_iteration > 0:
                break
