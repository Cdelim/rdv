import rdv
import torch


class ExperimentalGrid3D(rdv.Map):
    """
    Maps 3D coordinates x,y,z in range [-1, 1] to a regular grid of values
    at indices vz, vy, vx from a tenor (D, H, W, C).
    If align corner is true, the 0, 0, 0 is exactly the value 0,0,0 in the grid
    """
    __extension_info__ = dict(
        path="./experimental_grid3d.h", # The path to the file with the FORWARD and BACKWARD implementations
        parameters=dict(
            grid=torch.Tensor,  # inside code parameters.grid is a tensor
            shape=[3, int],  # inside code parameters.shape is a int[3]
            align_corners=int,  # inside code parameters.align_corners is an int (0 or 1)
        )
    )

    def __init__(self,
                 grid: rdv.TensorLike,
                 align_corners: bool | int = True,
                 # all maps must receive named arguments input_dim, output_dim, input_requires_grad and bw_uses_output
                 input_dim=3, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        # grid is converted to a tensor in case other types are passed
        grid = rdv.ensure_tensor(grid, map_dim=4)
        # if output_dim is None it can be inferred from the last dimension in the tensor
        if output_dim is None:
            output_dim = grid.shape[-1]
        # output_dim must match last dimension of the tensor
        assert output_dim == grid.shape[-1]
        # input_dim must be 3 (x, y, z)
        assert input_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
                         bw_uses_output=bw_uses_output)
        # bind arguments to parameters
        # from the __extension_info__['parameters'] dict there are attributes
        # handling from grid, shape and align_corners
        self.grid = grid
        # arrays must be set by indexing
        for i in range(3):
            self.shape[i] = grid.shape[i]
        # internally boolean is not supported using int instead
        self.align_corners = int(align_corners)

    def clone(self,
              **kwargs) -> rdv.Map:
        """
        This method is important to make all autocast mechanism to work.
        It just recreates the map with the same bound parameters, changing potentially
        input_dim, output_dim in **kwargs, but that's automatic.
        """
        return ExperimentalGrid3D(self.grid, self.align_corners, **kwargs)
