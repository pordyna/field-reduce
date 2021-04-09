import openpmd_api as io
import sys
import os.path
import time
import numpy as np
from typing import Sequence, Optional

def copy_attributes(source: io.Attributable,
                    target: io.Attributable) -> None:

    for attribute in source.attributes:
        target.set_attribute(attribute, source.get_attribute(attribute))


class OutputReducer:

    def __init__(self, sst_file_path: str, output_path: str, div_x: int,
                 div_y: int, div_z: int,
                 meshes: Optional[Sequence[str]] = None):
        if meshes is not None:
            self.meshes_to_reduce = meshes
        self.axis_scaling = {'x': div_x, 'y': div_y, 'z': div_z}
        self.output_series = io.Series(output_path, io.Access.create)
        while not os.path.exists(sst_file_path):
            time.sleep(10)
        self.input_series = io.Series(sst_file_path, io.Access.read_only)
        self.stored_meshes = {}

    def _process_mrc_before_close(self, input_mrc: io.Mesh_Record_Component,
                    output_mrc: io.Mesh_Record_Component, mesh_dict: dict, mrc_name: str) -> None:
        copy_attributes(input_mrc, output_mrc)
        self.output_series.flush()
        input_data = input_mrc.load_chunk()
        self.input_series.flush()
        mesh_dict[mrc_name] = input_data


    def _process_mesh_before_close(self, input_mesh: io.Mesh, output_mesh: io.Mesh, mesh_dict: dict)-> None:
        copy_attributes(input_mesh, output_mesh)
        new_grid_spacing = input_mesh.grid_spacing
        for dd in range(len(new_grid_spacing)):
            new_grid_spacing[dd] *= self.axis_scaling[input_mesh.axis_labels
            [dd]]
        output_mesh.set_grid_spacing(new_grid_spacing)
        self.output_series.flush()
        # go over record components:
        for mrc_name in input_mesh:
            self._process_mrc_before_close(input_mesh[mrc_name], output_mesh[mrc_name], mesh_dict, mrc_name),
        self.output_series.flush()

    def reduce(self):
        copy_attributes(self.input_series, self.output_series)
        self.output_series.flush()
        for input_iteration in self.input_series.read_iterations():
            # create iteration and copy attributes
            (output_iteration =
            self.output_series.iterations[input_iteration.iteration_index])
            copy_attributes(input_iteration, output_iteration)
            self.output_series.flush()

            # handle meshes
            input_meshes = input_iteration.meshes
            output_meshes = output_series.meshes
            copy_attributes(input_meshes, output_meshes)
            self.output_series.flush()

            for mesh_name in input_meshes:
                mesh_dict = self.stored_meshes['mesh_name'] = {}
                self._process_mesh_before_close(input_meshes[mesh_name], output_meshes[mesh_name], mesh_dict)
            input_iteration.close()




if __name__ == "__main__":
    if 'adios2' not in io.variants or not io.variants['adios2']:
        print('This requires ADIOS2')
        sys.exit(0)

    series = io.Series("stream.sst", io.Access_Type.read_only)

    backends = io.file_extensions
    if "sst" not in backends:
        print("SST engine not available in ADIOS2.")
        sys.exit(0)

    for iteration in series.read_iterations():
        print("Current iteration {}".format(iteration.iteration_index))
        electronPositions = iteration.particles["e"]["position"]
        loadedChunks = []
        shapes = []
        dimensions = ["x", "y", "z"]

        for i in range(3):
            dim = dimensions[i]
            rc = electronPositions[dim]
            loadedChunks.append(rc.load_chunk([0], rc.shape))
            shapes.append(rc.shape)
        iteration.close()

        for i in range(3):
            dim = dimensions[i]
            shape = shapes[i]
            print("dim: {}".format(dim))
            chunk = loadedChunks[i]
            print(chunk)
