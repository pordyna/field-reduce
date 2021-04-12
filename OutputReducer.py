import openpmd_api as io
import os.path
import time
import numpy as np
from typing import Optional, Iterable

from downscale_kernel import downscale


def copy_attributes(source: io.Attributable,
                    target: io.Attributable) -> None:
    dtypes = source.attribute_dtypes
    for attribute in source.attributes:
        target.set_attribute(attribute, source.get_attribute(attribute), dtypes[attribute])


class OutputReducer:

    def __init__(self, sst_file_path: str, output_path: str, div_x: int,
                 div_y: int, div_z: int,
                 meshes: Optional[Iterable[str]] = None, wait: bool = False, options_in = '{}', options_out = '{}'):
        if meshes is not None:
            self.meshes_to_reduce = meshes
        self.axis_scaling = {'x': div_x, 'y': div_y, 'z': div_z}
        self.output_series = io.Series(output_path, io.Access.create, options_out)
        if wait:
            while not os.path.exists(sst_file_path):
                time.sleep(10)
        self.input_series = io.Series(sst_file_path, io.Access.read_only, options_in)
        self.stored_meshes: dict[str, dict[str, np.ndarray]] = {}

    def _process_mrc_before_close(self, input_mrc: io.Mesh_Record_Component,
                                  output_mrc: io.Mesh_Record_Component, mesh_dict: dict, mrc_name: str) -> None:
        copy_attributes(input_mrc, output_mrc)
        input_data = input_mrc.load_chunk()
        self.input_series.flush()
        mesh_dict[mrc_name] = input_data

    def _process_mesh_before_close(self, input_mesh: io.Mesh, output_mesh: io.Mesh, mesh_dict: dict) -> None:
        copy_attributes(input_mesh, output_mesh)
        # go over record components:
        for mrc_name in input_mesh:
            self._process_mrc_before_close(input_mesh[mrc_name], output_mesh[mrc_name], mesh_dict, mrc_name),

    def reduce(self):
        copy_attributes(self.input_series, self.output_series)
        write_iterations = self.output_series.write_iterations()
        for input_iteration in self.input_series.read_iterations():
            # create iteration and copy attributes
            output_iteration = write_iterations[input_iteration.iteration_index]
            copy_attributes(input_iteration, output_iteration)
            self.output_series.flush()

            # handle meshes
            input_meshes = input_iteration.meshes
            output_meshes = output_iteration.meshes
            copy_attributes(input_meshes, output_meshes)
            self.output_series.flush()

            for mesh_name in input_meshes:
                mesh_dict = self.stored_meshes[mesh_name] = {}
                self._process_mesh_before_close(input_meshes[mesh_name], output_meshes[mesh_name], mesh_dict)
            input_iteration.close()

            for mesh_name, mesh_dict in self.stored_meshes.items():
                mesh = output_iteration.meshes[mesh_name]
                new_grid_spacing = mesh.grid_spacing
                if mesh_name in self.meshes_to_reduce or self.meshes_to_reduce is None:
                    for dd in range(len(new_grid_spacing)):
                        new_grid_spacing[dd] *= self.axis_scaling[mesh.axis_labels[dd]]
                    mesh.set_grid_spacing(new_grid_spacing)
                for mrc_name, mrc_data_old in self.stored_meshes[mesh_name].items():
                    if mesh_name in self.meshes_to_reduce or self.meshes_to_reduce is None:
                        new_shape = list(mrc_data_old.shape)
                        for dd in range(len(new_shape)):
                            new_shape[dd] = new_shape[dd] // self.axis_scaling[mesh.axis_labels[dd]]
                        mrc_data = np.empty(shape=new_shape, dtype=mrc_data_old.dtype)
                        downscale(mrc_data_old, mrc_data)
                    else:
                        mrc_data = mrc_data_old
                    mrc = mesh[mrc_name]
                    dataset = io.Dataset(mrc_data.dtype, mrc_data.shape)
                    mrc.reset_dataset(dataset)
                    mrc.store_chunk(mrc_data)
            output_iteration.close()
