import openpmd_api as api
from openpmd_api.pipe import FallbackMPICommunicator, Chunk
import os.path
import time
import numpy as np
from typing import Optional, Iterable, Tuple

try:
    from mpi4py import MPI
    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

from downscale_kernel import downscale


def copy_attributes(source: api.Attributable,
                    target: api.Attributable) -> None:
    dtypes = source.attribute_dtypes
    for attribute in source.attributes:
        target.set_attribute(attribute, source.get_attribute(attribute), dtypes[attribute])


class OutputReducer:

    def __init__(self, source_path: str, output_path: str, div_x: int,
                 div_y: int, div_z: int,
                 meshes: Optional[Iterable[str]] = None,
                 exclude: Optional[Iterable[str]] = None,
                 wait: bool = False, options_in='{}', options_out='{}'):
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
            print(f"You are using MPI. Welcome from rank {self.comm.rank} of {self.comm.size}")
        else:
            self.comm = FallbackMPICommunicator()
            print("In serial mode")
        self.output_series = api.Series(output_path, api.Access.create, self.comm, options_out)
        if wait:
            while not os.path.exists(source_path):
                time.sleep(10)
        self.input_series = api.Series(source_path, api.Access.read_only, self.comm, options_in)
        self.stored_meshes: dict[str, dict[str, Tuple[np.ndarray, Tuple]]] = {}

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
                                  output_mrc: api.Mesh_Record_Component, mesh_dict: dict, mrc_name: str) -> None:
        copy_attributes(input_mrc, output_mrc)
        shape = input_mrc.shape
        offset = [0 for _ in shape]
        chunk = Chunk(offset, shape)
        local_chunk = chunk.slice1D(self.comm.rank, self.comm.size)
        input_data = input_mrc.load_chunk(local_chunk.offset, local_chunk.extent)
        self.input_series.flush()
        mesh_dict[mrc_name] = (input_data, shape)

    def _process_mesh_before_close(self, input_mesh: api.Mesh, output_mesh: api.Mesh, mesh_dict: dict) -> None:
        copy_attributes(input_mesh, output_mesh)
        # go over record components:
        for mrc_name in input_mesh:
            self._process_mrc_before_close(input_mesh[mrc_name], output_mesh[mrc_name], mesh_dict, mrc_name),

    def run(self):
        copy_attributes(self.input_series, self.output_series)
        write_iterations = self.output_series.write_iterations()
        for input_iteration in self.input_series.read_iterations():
            idx = input_iteration.iteration_index
            print(f"[rank: {self.comm.rank}]:  Starting to process iteration number {idx}."
                  f" Starting to read data from source.")
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
                mesh_dict = self.stored_meshes[mesh_name] = {}
                self._process_mesh_before_close(input_meshes[mesh_name], output_meshes[mesh_name], mesh_dict)
            input_iteration.close()
            print(f"[rank: {self.comm.rank}]: Iteration number {idx} : All iteration data loaded from source. "
                  f"Now, processing and writing data.")
            for mesh_name, mesh_dict in self.stored_meshes.items():
                mesh = output_iteration.meshes[mesh_name]
                new_grid_spacing = mesh.grid_spacing
                if self._to_be_reduced(mesh_name):
                    for dd in range(len(new_grid_spacing)):
                        new_grid_spacing[dd] *= self.axis_scaling[mesh.axis_labels[dd]]
                    mesh.set_grid_spacing(new_grid_spacing)
                for mrc_name, (mrc_data_old, old_global_shape) in self.stored_meshes[mesh_name].items():
                    if self._to_be_reduced(mesh_name):
                        new_global_shape = list(old_global_shape)
                        for dd in range(len(new_global_shape)):
                            new_global_shape[dd] = new_global_shape[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                        offset = [0 for _ in new_global_shape]
                        global_chunk = Chunk(offset, new_global_shape)
                        local_chunk = global_chunk.slice1D(self.comm.rank, self.comm.size)
                        local_shape = list(local_chunk.extent)
                        for dd, extend in enumerate(local_shape):
                            local_shape[dd] = extend - local_chunk.offset[dd]
                            if mrc_data_old.shape[dd] % local_shape[dd] != 0:
                                raise ValueError(f"[rank: {self.comm.rank}, mesh: {mesh_name}, mrc: {mrc_name}]: "
                                                 f"Local output shape {local_shape} is not compatible with the local "
                                                 f"input shape {mrc_data_old.shape}. {local_shape[dd]} does not divide"
                                                 f" {mrc_data_old.shape[dd]}!")

                        mrc_data = np.empty(shape=local_shape, dtype=mrc_data_old.dtype)
                        downscale(mrc_data_old, mrc_data)
                    else:
                        offset = [0 for _ in old_global_shape]
                        global_chunk = Chunk(offset, old_global_shape)
                        local_chunk = global_chunk.slice1D(self.comm.rank, self.comm.size)
                        mrc_data = mrc_data_old
                    mrc = mesh[mrc_name]
                    dataset = api.Dataset(mrc_data.dtype, mrc_data.shape)
                    mrc.reset_dataset(dataset)
                    mrc.store_chunk(mrc_data, local_chunk.offset, local_chunk.extent)
            output_iteration.close()
            self.stored_meshes = {}
            print(f"[rank: {self.comm.rank}]: Finished processing iteration number {idx}.")
