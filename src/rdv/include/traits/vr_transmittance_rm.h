void transmittance_RM(MAP_DECL, vec3 x, vec3 w, float d, inout float T) {
    float t = 0.0;
    float tau = 0.0;
    while (t < d)
    {
        float density_val = density(_this, x + w * t);
        tau += density_val * parameters.step_size;
        t += parameters.step_size;
    }
    T *= exp(-tau);
}

void transmittance_RM_bw(MAP_DECL, vec3 x, vec3 w, float d, float final_T, float dL_dT) {
    float dL_dtau = -final_T * dL_dT;
    if (dL_dtau == 0.0)
        return;
    float dL_dsigma = dL_dtau * parameters.step_size;
    float t = 0.0;
    while (t < d)
    {
        vec3 pos = x + w * t;
        density_bw(_this, pos, dL_dsigma);
        t += parameters.step_size;
    }
}