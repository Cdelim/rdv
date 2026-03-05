/*
surfaces: Map ray -> surfel
patches: array of SPI_PatchInfo
SPI_PatchInfo:
    inside_medium: GPUPtr Map ray -> (A[OUTPUT_DIM], W[OUTPUT_DIM], x_o[3], w_o[3])
    outside_medium: GPUPtr Map ray -> (A[OUTPUT_DIM], W[OUTPUT_DIM], x_o[3], w_o[3])
    surface: GPUPtr Map ray + surfel -> (A[OUTPUT_DIM], W[OUTPUT_DIM], x_o[3], w_o[3])
environment: GPUPtr Map w -> (A[OUTPUT_DIM])
*/

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    for (int i = 0; i < OUTPUT_DIM; i++) _output[i] = 0.0; // A | PS
    float W[SPECTRAL_DIM]; // throughput along the path
    float temp[SPECTRAL_DIM*2 + 6 + PATH_STATE_DIM]; // A | W | x_o | w_o | PS
    float medium_input[7 + PATH_STATE_DIM]; // x (3), w (3), hit_t (1f), PS (PATH_STATE_DIM)
    float surface_input[6 + SURFEL_DIM + PATH_STATE_DIM];  // w_in (3), surfel (SURFEL_DIM), PS (PATH_STATE_DIM)
    float surfelInfo[SURFEL_DIM]; // surfel information
    for (int i = 0; i < SPECTRAL_DIM; i++) W[i] = 1.0;
    int medium_index = -1; // assume starting in unknown medium
    while(all(lessThan(abs(x), vec3(10000.0)))) // while the ray is valid
    {
        bool low_throughput = true;
        for (int i = 0; i < SPECTRAL_DIM; i++)
            if (W[i] > 0.0001)  // any throughput component is significant
            {
                low_throughput = false;
                break;
            }
        if (low_throughput)
            break;
        float hitInfo[5]; // path_index (int), hit_t (1f), surfel_code (3f)
        forward(parameters.surfaces, float[](x.x, x.y, x.z, w.x, w.y, w.z), hitInfo);
        int patch_index = floatBitsToInt(hitInfo[0]);
        if (patch_index == -1)
            break;
        // update medium index if in a supposed vacuum
        medium_index = medium_index >= 0 ? medium_index : hitInfo[1] < 0 ? parameters.inside_medium[patch_index] : parameters.outside_medium[patch_index];
        float hit_t = abs(hitInfo[1]); // signed distance to surface (positive for exterior and negative for interior)
        if (medium_index >= 0) {
            medium_input[0] = x.x;
            medium_input[1] = x.y;
            medium_input[2] = x.z;
            medium_input[3] = w.x;
            medium_input[4] = w.y;
            medium_input[5] = w.z;
            medium_input[6] = hit_t;
            for (int i = 0; i < PATH_STATE_DIM; i++)
                medium_input[7 + i] = _output[SPECTRAL_DIM + i];
            forward(parameters.medium_integrators[medium_index], medium_input, temp);
            // temp (A | W | x_o | w_o | PS)
            for (int i = 0; i < SPECTRAL_DIM; i++)
                _output[i] += W[i] * temp[i];
            for (int i = 0; i < SPECTRAL_DIM; i++)
                W[i] *= temp[SPECTRAL_DIM + i];
            vec3 xo = vec3(temp[SPECTRAL_DIM * 2 + 0], temp[SPECTRAL_DIM * 2 + 1], temp[SPECTRAL_DIM * 2 + 2]);
            vec3 wo = vec3(temp[SPECTRAL_DIM * 2 + 3], temp[SPECTRAL_DIM * 2 + 4], temp[SPECTRAL_DIM * 2 + 5]);
            for (int i = 0; i < PATH_STATE_DIM; i++)
                _output[SPECTRAL_DIM + i] = temp[SPECTRAL_DIM * 2 + 6 + i];
            // direction changed == some scatter in the medium
            if (any(notEqual(wo, w)))
            {
                x = xo;
                w = wo;
                continue; // it is still in the medium, continue the main loop
            }
        }

        // no medium interaction continue to surface
        // arrived to a surface
        x += w * hit_t;

        int surface_integrator_index = parameters.surface_integrator_indices[patch_index];
        if (surface_integrator_index >= 0)
        {
            forward(parameters.surfels, hitInfo, surfelInfo);
            surface_input[0] = x.x;
            surface_input[1] = x.y;
            surface_input[2] = x.z;
            surface_input[3] = w.x;
            surface_input[4] = w.y;
            surface_input[5] = w.z;
            for (int i = 0; i < SURFEL_DIM; i++)
                surface_input[6 + i] = surfelInfo[i];
            for (int i = 0; i < PATH_STATE_DIM; i++)
                surface_input[6 + SURFEL_DIM + i] = _output[SPECTRAL_DIM + i];
            forward(parameters.surface_integrators[surface_integrator_index], surface_input, temp);
            for (int i = 0; i < SPECTRAL_DIM; i++)
                _output[i] += W[i] * temp[i];
            for (int i = 0; i < SPECTRAL_DIM; i++)
                W[i] *= temp[SPECTRAL_DIM + i];
            for (int i = 0; i < PATH_STATE_DIM; i++)
                _output[SPECTRAL_DIM + i] = temp[SPECTRAL_DIM * 2 + 6 + i];
            w = vec3(temp[SPECTRAL_DIM * 2 + 3], temp[SPECTRAL_DIM * 2 + 4], temp[SPECTRAL_DIM * 2 + 5]);
            x = vec3(temp[SPECTRAL_DIM * 2 + 0], temp[SPECTRAL_DIM * 2 + 1], temp[SPECTRAL_DIM * 2 + 2]);
        }

        x += w * 0.0001; // advance the ray through the surface

        // check current medium
        if (hitInfo[0] < 0.0) // exiting the surface
            // only change if we are exiting the current medium
            medium_index = medium_index == parameters.inside_medium[patch_index] ? parameters.outside_medium[patch_index] : medium_index;
        else
            medium_index = medium_index == parameters.outside_medium[patch_index] ? parameters.inside_medium[patch_index] : medium_index;
    }
    // add environment contribution
    #ifdef HAS_ENVIRONMENT
    float eO[SPECTRAL_DIM]; // environment radiance
    forward(parameters.environment, float[](w.x, w.y, w.z), eO);
    for (int i = 0; i < SPECTRAL_DIM; i++)
        _output[i] += W[i] * eO[i];
    #endif
}