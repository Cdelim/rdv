float transmittance_rm(MAP_DECL, vec3 x, vec3 w, float d, float step_size)
{
    float tau = 0.0;
    x += step_size * w * random(); // start from the middle of the first step for better accuracy
    while (d > 0.) {
        float current_extinction = extinction(_this, x);
        tau += current_extinction * step_size; // trapezoidal rule for better accuracy
        if (tau > 20) return 0.0;
        d -= step_size;
        x += step_size * w;
    }
    return exp(-tau);
}

float transmittance_rm(MAP_DECL, vec3 x, vec3 w, float d, float step_size, float scale)
{
    float tau = 0.0;
    x += step_size * w * random(); // start from the middle of the first step for better accuracy
    while (d > 0.) {
        float current_extinction = extinction(_this, x);
        tau += current_extinction * step_size; // trapezoidal rule for better accuracy
        if (tau * scale > 20) return 0.0;
        d -= step_size;
        x += step_size * w;
    }
    return exp(-tau * scale);
}
