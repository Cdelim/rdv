//void blend(MAP_DECL, inout float dst[OUTPUT_DIM], float_ptr src, float alpha)
//{
//    /*[[unroll]]*/ for (int i=0; i<OUTPUT_DIM; i++) dst[i] += src.data[i] * alpha;
//}
//
//FORWARD
//{
//    vec3 c = vec3(_input[0], _input[1], _input[2]);
//    vec3 ncoord = c * 0.5 + 0.5;
//    /*[[unroll]]*/
//    for (int i=0; i<OUTPUT_DIM; i++)
//        _output[i] = 0.0;
//    if (any(lessThan(ncoord, vec3(0.0))) || any(greaterThanEqual(ncoord, vec3(1.0))))
//        return;
//    ivec3 dim = ivec3(parameters.shape[2] - 1, parameters.shape[1] - 1, parameters.shape[0] - 1);
//    vec3 grid_coord = ncoord * vec3(dim);
//    vec3 alpha = fract(grid_coord);
//    ivec3 p = clamp(ivec3(grid_coord), ivec3(0), dim - 1);
//    GPUPtr grid_ptr = load_tensor(parameters.grid);
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(0,0,0)), (1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z));
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(1,0,0)), alpha.x*(1 - alpha.y)*(1 - alpha.z));
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(0,1,0)), (1 - alpha.x)*alpha.y*(1 - alpha.z));
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(1,1,0)), alpha.x*alpha.y*(1 - alpha.z));
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(0,0,1)), (1 - alpha.x)*(1 - alpha.y)*alpha.z);
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(1,0,1)), alpha.x*(1 - alpha.y)*alpha.z);
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(0,1,1)), (1 - alpha.x)*alpha.y*alpha.z);
//    blend(_this, _output, tensor_at(grid_ptr, OUTPUT_DIM, parameters.shape, p + ivec3(1,1,1)), alpha.x*alpha.y*alpha.z);
//}



/* Parameters
grid: deferred
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