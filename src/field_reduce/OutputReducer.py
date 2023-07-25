import openpmd_api as api
#from openpmd_api.pipe.__main__ import FallbackMPICommunicator, Chunk
import os.path
import time
import numpy as np
import time

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
    dtypes = source.attribute_dtypes
    for attribute in source.attributes:
        if attribute == "iterationEncoding":
            continue
        if attribute == "iterationFormat":
            continue
        target.set_attribute(attribute, source.get_attribute(attribute), dtypes[attribute])


class OutputReducer:

    def __init__(self, source_path: str, output_path: str, div_x: int,
                 div_y: int, div_z: int,
                 meshes: Optional[Iterable[str]] = None,
                 exclude: Optional[Iterable[str]] = None,
                 wait: bool = False, options_in='{}', options_out='{}', last_iteration=-1, first_iteration=-1):
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
                                  output_mrc: api.Mesh_Record_Component, mesh_dict: dict, mrc_name: str, divisible: int) -> None:
        copy_attributes(input_mrc, output_mrc)
        shape = input_mrc.shape
        offset = [0 for _ in shape]
        chunk = Chunk(offset, shape)
        local_chunk = chunk.slice1D(self.comm.rank, self.comm.size, dimension=0, divisible=divisible)
        input_data = input_mrc.load_chunk(local_chunk.offset, local_chunk.extent)
        # self.input_series.flush()
        mesh_dict[mrc_name] = (input_data, shape)

    def _process_mesh_before_close(self, input_mesh: api.Mesh, output_mesh: api.Mesh, mesh_dict: dict, divisible:bool = False) -> None:
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
            copy_attributes(input_iteration, output_iteration)
            # self.output_series.flush()

            # handle meshes
            input_meshes = input_iteration.meshes
            output_meshes = output_iteration.meshes
            copy_attributes(input_meshes, output_meshes)
            # self.output_series.flush()

            for mesh_name in input_meshes:
                to_be_reduced = self._to_be_reduced(mesh_name)
                mesh_dict = self.stored_meshes[mesh_name] = {}
                self._process_mesh_before_close(input_meshes[mesh_name], output_meshes[mesh_name], mesh_dict, divisible=to_be_reduced)
                # input_iteration.close()
                # print(f"[rank: {self.comm.rank}]: Iteration number {idx} : All iteration data loaded from source. "
                #      f"Now, processing and writing data.")
                self.input_series.flush()
                mesh = output_iteration.meshes[mesh_name]
                new_grid_spacing = mesh.grid_spacing
                if to_be_reduced:
                    for dd in range(len(new_grid_spacing)):
                        new_grid_spacing[dd] *= self.axis_scaling[mesh.axis_labels[dd]]
                    mesh.set_grid_spacing(new_grid_spacing)
                for mrc_name, (mrc_data_old, old_global_shape) in self.stored_meshes[mesh_name].items():
                    if to_be_reduced:
                        offset = [0 for _ in old_global_shape]
                        old_global_chunk = Chunk(offset, old_global_shape)
                        old_local_chunk = old_global_chunk.slice1D(self.comm.rank, self.comm.size,
                         dimension=0, divisible=self.axis_scaling[mesh.axis_labels[0]])
                        local_chunk = old_local_chunk
                        global_chunk = old_global_chunk
                        for dd in range(len(old_global_shape)):
                            local_chunk.extent[dd] =local_chunk.extent[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                            local_chunk.offset[dd] =local_chunk.offset[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                            global_chunk.extent[dd] = global_chunk.extent[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                            global_chunk.offset[dd] = global_chunk.offset[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                        #for dd, extent in enumerate(local_chunk.extent):
                            # if mrc_data_old.shape[dd] % extent != 0:
                            #     raise ValueError(f"[rank: {self.comm.rank}, mesh: {mesh_name}, mrc: {mrc_name}]: "
                            #                      f"Local output shape {local_chunk.extent} is not compatible with the "
                            #                      f"local input shape {mrc_data_old.shape}. {local_chunk.extent[dd]} "
                            #                      f"does not divide {mrc_data_old.shape[dd]}!")

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
            print(f"[rank: {self.comm.rank}]: Finished processing iteration number {idx}. Took {elapsed//60} m {elapsed % 60} s", flush=True)
            if idx >= self.last_iteration > 0:
                break
