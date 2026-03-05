/* Parameters
grid: deferred
*/

void scanline_interpolator(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, float_ptr src_right, float x_weight) {
    for (int i=0; i<OUTPUT_DIM; i++)
        dst[i] += mix(src_left.data[i], src_right.data[i], x_weight);
}

FORWARD {
    Tensor grid = load_deferred(parameters.grid);
    for(int i=0; i<OUTPUT_DIM; i++) _output[i] = 0.0;
    float coord = _input[0]*0.5 + 0.5;
    float grid_size = float(grid.shape[0]);
    float index_f = parameters.align_corners != 0 ? coord * (grid_size - 1.0) : coord * grid_size - 0.5;
    int index0 = int(floor(index_f));
    int index1 = index0 + 1;
    float alpha = index_f - float(index0);
    index0 = clamp(index0, 0, int(grid_size) - 1);
    index1 = clamp(index1, 0, int(grid_size) - 1);
    int stride_x = OUTPUT_DIM * 4;
    GPUPtr ptr = grid.data_ptr + index0.x * stride_x;
    stride_x *= (index1.x - index0.x);
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x);
}

void scanline_interpolator_bw(MAP_DECL, float_ptr dst_a_grad, float_ptr dst_b_grad,
in float _output_grad[OUTPUT_DIM], float x_weight) {
    for (int i=0; i<OUTPUT_DIM; i++) {
        atomicAdd_f(dst_a_grad, i, (1 - x_weight) * _output_grad[i]);
        atomicAdd_f(dst_b_grad, i, x_weight * _output_grad[i]);
    }
}

BACKWARD {
    Tensor grid = load_deferred(parameters.grid);
#ifndef INPUT_REQUIRES_GRAD
    if (grid.grad_ptr == 0) return;  // no gradient needed
#else
    NOT_SUPPORTED("Input requires grad");
#endif
    float coord = _input[0]*0.5 + 0.5;
    float grid_size = float(grid.shape[0]);
    float index_f = parameters.align_corners != 0 ? coord * (grid_size - 1.0) : coord * grid_size - 0.5;
    int index0 = int(floor(index_f));
    int index1 = index0 + 1;
    float alpha = index_f - float(index0);
    index0 = clamp(index0, 0, int(grid_size) - 1);
    index1 = clamp(index1, 0, int(grid_size) - 1);
    int stride_x = OUTPUT_DIM * 4;
    GPUPtr ptr = grid.grad_ptr + index0.x * stride_x;
    stride_x *= (index1.x - index0.x);
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x);
}