FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery,              // Ray query
                        accelerationStructureEXT(parameters.scene_ads),                  // Top-level acceleration structure
                        gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                        0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                        x,                  // Ray origin
                        0.0,                   // Minimum t-value
                        w,            // Ray direction
                        10000.0);              // Maximum t-value

    while (rayQueryProceedEXT(rayQuery)); // traverse to find intersections
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
    {
        int index = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true); // instance index
        int primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
        vec2 baricentrics = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
        float t = rayQueryGetIntersectionTEXT(rayQuery, true);
        _output = float[](
            intBitsToFloat(index),
            t,
            intBitsToFloat(primitive_index),
            baricentrics.x, baricentrics.y
        );
    }
    else { // miss geometries
        _output = float[](
            intBitsToFloat(-1),
            POSINF,
            intBitsToFloat(-1),
            0.0, 0.0
        );
    }
}

BACKWARD {

}