vec3 SUBMAP_FORWARD_NAME(MAP_DECL, vec3 x) {
    float[SPECTRAL_DIM] o;
    forward(parameters.SUBMAP_NAME, float[3](x.x, x.y, x.z), o);
    return vec3(o[0], o[1], o[2]);
}



#undef SUBMAP_NAME // make it here
#undef SUBMAP_BW_USES_OUTPUT
#undef SUBMAP_INPUT_REQUIRES_GRAD
