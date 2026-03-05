FORWARD {
#if INPUT_DIM == 3
    vec3 x = vec3(_input[0], _input[1], _input[2]);
#elif INPUT_DIM == 2
    vec2 x = vec2(_input[0], _input[1]);
#elif INPUT_DIM == 1
    float x = _input[0];
#endif

x = x * 0.5 + 0.5; // transform from [-1, 1] to [0, 1], since texture sampling is in [0, 1]

if (parameters.align_corners != 0) {
    for (int i=0; i<INPUT_DIM; i++)
        x[i] = x[i] * (parameters.shape[INPUT_DIM - 1 - i] - 1.0) / parameters.shape[INPUT_DIM - 1 - i] + 0.5 / parameters.shape[INPUT_DIM - 1 - i];
}

#if INPUT_DIM == 3
    vec4 v = texture(
        sampler3D(
            rdv_textures_3D[parameters.image_index],
            rdv_samplers[parameters.sampler_index]), x);
#elif INPUT_DIM == 2
    vec4 v = texture(
        sampler2D(
            rdv_textures_2D[parameters.image_index],
            rdv_samplers[parameters.sampler_index]), x);
#elif INPUT_DIM == 1
    vec4 v = texture(
        sampler1D(
            rdv_textures_1D[parameters.image_index],
            rdv_samplers[parameters.sampler_index]), x);
#endif

    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = v[i];
}