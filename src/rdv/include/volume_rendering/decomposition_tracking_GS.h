/*
This map receives a ads data-structure with AABB for each gaussian
parameters:
- ads: Data structure to query the index of the AABB
- positions: each gaussian's position
- cov: each gaussian's covariance
- scale: each gaussian's scale
- sh: each gaussian's spherical harmonic coefficients
*/

FORWARD {
    GPUPtr positions_ptr = load_tensor(parameters.positions);
    vec3_ptr positions = vec3_ptr(positions_ptr);

    GPUPtr colors_ptr = load_tensor(parameters.colors);
    vec3_ptr colors = vec3_ptr(colors_ptr);

    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    float T = 1.0; // transmittance
    vec3 A = vec3(0.0);
    float current_t = 0;
    rayQueryEXT rayQuery;
    while (true) {
        rayQueryInitializeEXT(rayQuery,              // Ray query
                            accelerationStructureEXT(parameters.ads),                  // Top-level acceleration structure
                            gl_RayFlagsOpaqueEXT ,  // Ray flags, here saying "treat all geometry as opaque"
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            x,                  // Ray origin
                            current_t + 0.0001,                   // Minimum t-value
                            w,            // Ray direction
                            10000.0);              // Maximum t-value

        // you need to care for the minimum t-value when traversing the ray query, otherwise you might get stuck at the same intersection and cause infinite loop
        float minimum_t = 10000.0;
        while(rayQueryProceedEXT(rayQuery)) {
            if (rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionAABBEXT)
            {
                // get the primitive index (AABB index)
                int primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
                float tmin, tmax;
                ray_sphere_intersection(x, w, positions.data[primitive_index], parameters.radius, tmin, tmax);
                if (tmax <= 0.0 || tmin >= tmax || current_t >= tmax) // no intersection with the AABB after current_t
                    continue;
                float t = tmin <= current_t ? tmax : tmin; // if tmin is negative, it means the ray origin is inside the box, we take tmax as the exit point
                // Report intersection candidate
                if (t < minimum_t) {
                    minimum_t = t;
                    rayQueryGenerateIntersectionEXT(rayQuery, t);
                }
            }
        }

        if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == 0)
        break;

        // get the committed intersection information
        int primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
        current_t = rayQueryGetIntersectionTEXT(rayQuery, true) + 0.001;
        float alpha = 0.85;
        A += T * alpha * colors.data[primitive_index]; // some color there...
        T *= (1 - alpha); // opacity loss
        if (T < 0.01)
        break;
    }
    _output = float[](A.x, A.y, A.z); // finish
}

BACKWARD {
}