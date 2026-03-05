#define MAIN(tid) void rdv_compute(uvec3 tid)

void rdv_compute(uvec3 thread_id);

#include "core.h"

layout (local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

layout(set=0, binding=0) uniform rdv_SystemInfo {
    uvec4 rdv_seeds;
    // total dimensions (not necessary the dispatch size if batched)
    int rdv_dim_x;
    int rdv_dim_y;
    int rdv_dim_z;
    // current batch offset
    int rdv_start_x;
    int rdv_start_y;
    int rdv_start_z;
};

layout(set=0, binding=3) uniform sampler rdv_samplers[2];
layout(set=1, binding=4) uniform texture1D rdv_textures_1D[1024];
layout(set=1, binding=5) uniform texture2D rdv_textures_2D[1024];
layout(set=1, binding=6) uniform texture3D rdv_textures_3D[1024];

void main()
{
    uvec3 global_id = gl_GlobalInvocationID;
    global_id += uvec3(rdv_start_x, rdv_start_y, rdv_start_z);
    if (any(greaterThanEqual(global_id, uvec3(rdv_dim_x, rdv_dim_y, rdv_dim_z))))
        return;
    #ifdef RDV_STOCHASTIC_COMPUTE
    int index = int(global_id.x + (global_id.z * rdv_dim_y + global_id.y) * rdv_dim_x);
    random_seed(random_spawn(rdv_seeds, index));
    #endif
    rdv_compute(global_id);
}

