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
    float T = 1.0;
    int medium_index = -1; // assume starting in unknown medium
    while(all(lessThan(abs(x), vec3(10000.0)))) // while the ray is valid
    {
        if (T < 0.01)
        {
            if (random() >= T)
            {
                T = 0.0;
                break;
            }
            T = 1.0;
        }

        float hitInfo[5]; // path_index (int), hit_t (1f), surfel_code (3f)
        forward(parameters.surfaces, float[](x.x, x.y, x.z, w.x, w.y, w.z), hitInfo);
        int patch_index = floatBitsToInt(hitInfo[0]);
        if (patch_index == -1)
            break;
        if (parameters.has_surface[patch_index] > 0)
        {
            T = 0;
            break;
        }
        medium_index = medium_index >= 0 ? medium_index : hitInfo[1] < 0 ? parameters.inside_medium[patch_index] : parameters.outside_medium[patch_index];
        float hit_t = abs(hitInfo[1]); // signed distance to surface (positive for exterior and negative for interior)
        // update medium index if in a supposed vacuum
        if (medium_index >= 0) {
            float mT[1]; // medium transmittance
            forward(parameters.medium_transmittance[medium_index], float[](x.x, x.y, x.z, w.x, w.y, w.z, hit_t), mT);
            T *= mT[0];
        }

        // no medium interaction continue to surface
        // arrived to a null surface
        x += w * (hit_t + 0.0001); // advance the ray through the surface

        // check current medium
        if (hitInfo[1] < 0) // exiting the surface
            // only change if we are exiting the current medium
            medium_index = medium_index == parameters.inside_medium[patch_index] ? parameters.outside_medium[patch_index] : medium_index;
        else
            medium_index = medium_index == parameters.outside_medium[patch_index] ? parameters.inside_medium[patch_index] : medium_index;
    }
    _output[0] = T;
}