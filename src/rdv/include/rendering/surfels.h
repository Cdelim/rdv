FORWARD {
    int index = floatBitsToInt(_input[0]);
    float hit_t = _input[1];
    if (index == -1) {
        for (int i = 0; i < OUTPUT_DIM; i++)
            _output[i] = 0;
        return;
    }
    int primitive_index = floatBitsToInt(_input[2]);
    vec2 baricentrics = vec2(_input[3], _input[4]);
    mat4x3 o2w = mat4x3_ptr(load_tensor(parameters.transforms[index])).data[0];
    vec3 alphas = vec3(1 - baricentrics.x - baricentrics.y, baricentrics.x, baricentrics.y);
    int_ptr indices = int_ptr(parameters.patches[index].indices);
    int idx0 = indices.data[primitive_index * 3 + 0];
    int idx1 = indices.data[primitive_index * 3 + 1];
    int idx2 = indices.data[primitive_index * 3 + 2];
    vec3_ptr pos = vec3_ptr(load_tensor(parameters.patches[index].positions));
    #ifdef POSITION_INDEX
    vec3 P = pos.data[idx0] * alphas.x + pos.data[idx1] * alphas.y + pos.data[idx2] * alphas.z;
    P = transform_position(P, o2w);
    _output[POSITION_INDEX] = P.x;
    _output[POSITION_INDEX + 1] = P.y;
    _output[POSITION_INDEX + 2] = P.z;
    #endif
    vec3 Nface = normalize(cross(pos.data[idx1] - pos.data[idx0], pos.data[idx2] - pos.data[idx0]));
    #ifdef GEOMETRY_NORMAL_INDEX
    vec3 Ng = transform_normal(Nface, o2w);
    _output[GEOMETRY_NORMAL_INDEX] = Ng.x;
    _output[GEOMETRY_NORMAL_INDEX + 1] = Ng.y;
    _output[GEOMETRY_NORMAL_INDEX + 2] = Ng.z;
    #endif
    #if defined(SHADING_NORMAL_INDEX) || defined(TANGENT_INDEX)
    GPUPtr normals_ptr = load_tensor(parameters.patches[index].normals);
    vec3_ptr nor = vec3_ptr(normals_ptr);
    vec3 N = normals_ptr == 0 ? Nface : normalize(nor.data[idx0] * alphas.x + nor.data[idx1] * alphas.y + nor.data[idx2] * alphas.z);
    N = transform_normal(N, o2w);
    #endif
    #ifdef SHADING_NORMAL_INDEX
    _output[SHADING_NORMAL_INDEX] = N.x;
    _output[SHADING_NORMAL_INDEX + 1] = N.y;
    _output[SHADING_NORMAL_INDEX + 2] = N.z;
    #endif
    #ifdef UV_INDEX
    GPUPtr coordinates_ptr = load_tensor(parameters.patches[index].uvs);
    vec2_ptr uvs = vec2_ptr(coordinates_ptr);
    vec2 C = coordinates_ptr == 0 ? vec2(0.0) : uvs.data[idx0] * alphas.x + uvs.data[idx1] * alphas.y + uvs.data[idx2] * alphas.z;
    _output[UV_INDEX] = C.x;
    _output[UV_INDEX + 1] = C.y;
    #endif
    #ifdef TANGENT_INDEX
    //Tensor tangents = load_deferred(parameters.patches[index].tangents);
    //vec3_ptr tan = vec3_ptr(tangents.data_ptr);
    //vec3 T = tangents.data_ptr == 0 ? (1.0 - abs(NFace.x) <= 0.9 ? vec3(1, 0, 0): vec3(0, 1, 0)) : normalize(tan.data[idx0] * alphas.x + tan.data[idx1] * alphas.y + tan.data[idx2] * alphas.z);
    vec3 T = 1.0 - abs(N.x) <= 0.01 ? vec3(0.0, 1.0, 0.0): vec3(1.0, 0.0, 0.0);
    T = normalize(cross(N, T));
    _output[TANGENT_INDEX] = T.x;
    _output[TANGENT_INDEX + 1] = T.y;
    _output[TANGENT_INDEX + 2] = T.z;
    #endif
    // TODO: tangent vector from mesh if available or computed from dP / du
}

BACKWARD {

}