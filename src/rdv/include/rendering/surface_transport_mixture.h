/*
    num_weight_maps: int[num_materials, num_repeats]
    weight_maps: map[num_materials, num_repeats, max_num_weight_maps]
    lobes: [num_layers, num_materials]
    num_materials: int[num_layers]
    num_layers: int
*/

float compute_weight(MAP_DECL, int material_type, int index, vec2 C){
    float w = 1.0;
    for (int i = 0; i<parameters.num_weight_maps[material_type, index]; i++) {
        float map_value[1];
        forward(parameters.weight_maps[material, i], float[](C.x, C.y), map_value);
        w *= map_value[0];
    }
    return w;
}

FORWARD {
    int current_layer = 0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] = 1.0; // initialize throughput
    while (current_layer >= 0 && current_layer < parameters.num_layers) {
        int material = parameters.lobes[current_layer];
        vec2 C = vec2(_input[7], _input[8]);
        float weight = compute_weight(MAP_PASS, current_layer, material, C);
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i] *= weight;
        // determine next layer
        if (weight < 0.01) {
            current_layer = -1; // terminate
        } else {
            current_layer += 1;
            if (current_layer >= parameters.num_layers) {
                current_layer = -1; // terminate
            }
        }
    }

}