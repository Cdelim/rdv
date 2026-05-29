/* Parameters
grid: tensor of shape (D, H, W, OUTPUT_DIM)
shape: int[3] with D, H, W
align_corners: int, whether the tensor grid represent corner values or voxel values
*/

void scanline_interpolator(MAP_DECL, inout float dst[OUTPUT_DIM],
    float_ptr src_left, float_ptr src_right, float x_weight, float yz_weight) {
    for (int i=0; i<OUTPUT_DIM; i++)
        dst[i] += yz_weight * mix(src_left.data[i], src_right.data[i], x_weight);
}

FORWARD {
    for(int i=0; i<OUTPUT_DIM; i++) _output[i] = 0.0;
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));
    //vec3 coord = vec3(_input[0], _input[1], _input[2]) * 0.5 + 0.5;
    //vec3 index_f = parameters.align_corners != 0 ? coord * (grid_size - vec3(1.0)) : coord * grid_size - vec3(0.5);
    vec3 index_f = (vec3(_input[0], _input[1], _input[2]) * (grid_size - parameters.align_corners) + grid_size) * 0.5 - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));
    int stride_x = OUTPUT_DIM * 4;
    int stride_y = int(parameters.shape[2]) * stride_x;
    int stride_z = int(parameters.shape[1]) * stride_y;
    GPUPtr ptr = load_tensor(parameters.grid) + index0.x * stride_x + index0.y * stride_y + index0.z * stride_z;
    stride_x *= (index1.x - index0.x);
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, (1 - alpha.y)*(1 - alpha.z));
    stride_y *= (index1.y - index0.y);
    ptr += stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, alpha.y * (1 - alpha.z));
    stride_z *= (index1.z - index0.z);
    ptr += stride_z - stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, (1 - alpha.y)*alpha.z);
    ptr += stride_y;
    scanline_interpolator(_this, _output,
        float_ptr(ptr),
        float_ptr(ptr + stride_x),
        alpha.x, alpha.y*alpha.z);
}

void scanline_interpolator_bw(MAP_DECL, float_ptr dst_a_grad, float_ptr dst_b_grad,
in float _output_grad[OUTPUT_DIM], float x_weight, float yz_weight) {
    for (int i=0; i<OUTPUT_DIM; i++) {
        atomicAdd_f(dst_a_grad, i, yz_weight * (1 - x_weight) * _output_grad[i]);
        atomicAdd_f(dst_b_grad, i, yz_weight * x_weight * _output_grad[i]);
    }
}

BACKWARD {
    GPUPtr ptr = load_tensor_grad(parameters.grid);
#ifndef INPUT_REQUIRES_GRAD
    if (ptr == 0) return;  // no gradient needed
#else
    NOT_SUPPORTED("Input requires grad");
#endif
    vec3 coord = vec3(_input[0], _input[1], _input[2])*0.5 + 0.5;
    vec3 grid_size = vec3(float(parameters.shape[2]), float(parameters.shape[1]), float(parameters.shape[0]));
    vec3 index_f = parameters.align_corners != 0 ? coord * (grid_size - vec3(1.0)) : coord * grid_size - vec3(0.5);
    ivec3 index0 = ivec3(floor(index_f));
    ivec3 index1 = index0 + ivec3(1);
    vec3 alpha = index_f - vec3(index0);
    index0 = clamp(index0, ivec3(0), ivec3(grid_size) - ivec3(1));
    index1 = clamp(index1, ivec3(0), ivec3(grid_size) - ivec3(1));
    int stride_x = OUTPUT_DIM * 4;
    int stride_y = int(parameters.shape[2]) * stride_x;
    int stride_z = int(parameters.shape[1]) * stride_y;
    ptr += index0.x * stride_x + index0.y * stride_y + index0.z * stride_z;
    stride_x *= (index1.x - index0.x);
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x, (1 - alpha.y)*(1 - alpha.z));
    stride_y *= (index1.y - index0.y);
    ptr += stride_y;
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x, alpha.y*(1 - alpha.z));
    stride_z *= (index1.z - index0.z);
    ptr += stride_z - stride_y;
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x, (1 - alpha.y)*alpha.z);
    ptr += stride_y;
    scanline_interpolator_bw(_this,
        float_ptr(ptr),
        float_ptr(ptr + stride_x), _output_grad,
        alpha.x, alpha.y*alpha.z);
}