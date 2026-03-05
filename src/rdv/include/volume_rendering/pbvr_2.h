#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"


float transmittance_rt(MAP_DECL, vec3 x, vec3 w)
{
    float tMin, d;
    ray_box_intersection(x, w, vec3(-1.0), vec3(1.0), tMin, d); // compute intersection with volume boundary for next iteration

    if (tMin > d) // ray points away from the volume
        return 1.0;

    float T = 1.0;
    while (d > 0.) {
        float maj_distance;
        float maj = majorant(_this, x, w, maj_distance);
        float dt = min(maj_distance, -log(1.0 - random()) / maj); // sample free-flight distance
        x += dt * w;
        d -= dt;
        if (dt == maj_distance)
            continue; // no interaction withing slab, continue

        if (d <= 0.0)
            break; // reached the end of the segment

        T *= (1.0 - extinction(_this, x) / maj);

        if (T < 0.01)
        {
            if (random() >= T)
                return 0.0;
            T = 1.0; // Russian roulette
        }
    }
    return T;
}


#define SUBMAP_NAME scattering_albedo
#include "signatures/vec3_to_vec3.h"

#define SUBMAP_NAME environment
#include "signatures/vec3_to_vec3.h"


FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    /*
    size = (bmax - bmin)
    scale = 2.0 / size
    translate = -(bmax + bmin)/size
    */

    // Transform ray to local space, from bmin,bmax to [-1,1]^3
    vec3 scale = vec3(2.0) / (parameters.bmax - parameters.bmin); // from [bmin,bmax] to [-1,1]]
    vec3 offset = -(parameters.bmax + parameters.bmin) / (parameters.bmax - parameters.bmin);
    x = x * scale + offset;
    vec3 wo = w * scale; // only scale direction, no translation
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function
    float tMin, tMax;
    ray_box_intersection(x, wo, vec3(-1.0), vec3(1.0), tMin, tMax);
    vec3 A = vec3(0.0);
    if (tMax <= 0 || tMin > tMax) // ray points away from the volume
    {
#ifdef ENVIRONMENT
        A += environment(_this, w); // direct environment lighting
#endif
        _output = float[3](A.x, A.y, A.z);
        return;
    }
    x += wo * tMin; // initial position in object space
    float d = tMax - tMin; // max_t wrt wo

    vec3 W = vec3(1.0); // path throughput, accumulation is direct to output to save register, so we need to store it in a local variable

    int bounces = 0;

    while (d > 0) {

        bounces ++;

        float maj_distance;
        float maj = majorant(_this, x, wo, maj_distance);
        float dt = min(maj_distance, -log(1.0 - random()) / maj); // sample free-flight distance
        x += dt * wo;
        d -= dt;
        if (dt == maj_distance)
            continue; // no interaction withing slab, continue
        if (d <= 0.0)
            break; // reached the end of the segment
        if (random() >= extinction(_this, x) / maj) // null collision
            continue;

        // -- Valid collision --
        // modulate path-throughput with scattering albedo
        W *= scattering_albedo(_this, x); // compute scattering albedo

        if (all(lessThan(W, vec3(0.0001))))
        {
            //A = complexity_color(bounces);
            _output = float[3](A.x, A.y, A.z);
            return; // early termination if path throughput is too low
        }

        // get anisotropy for phase function sampling
        float g = anisotropy(_this, x);

#ifdef ENVIRONMENT_SAMPLER
        // add contribution from environment lighting with NEE
        float env_sample[3 + SPECTRAL_DIM + 1]; // wnee, env/pdf(wnee), pdf (pdf not necessary here)
        // x is in object space, w is in world space, we need to convert x to world space for querying environment sampler, and convert nee_w back to object space for computing transmittance
        forward(parameters.environment_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), env_sample);
        vec3 nee_w = vec3(env_sample[0], env_sample[1], env_sample[2]);
        float nee_phase = hg_phase_eval(w, nee_w, g);
        nee_w *= scale; // convert to object space
        float tr = transmittance_rt(_this, x, nee_w);
         A += tr * W * vec3(env_sample[3], env_sample[4], env_sample[5]) * nee_phase; // accumulate NEE contribution
//        A += tr * W * vec3(3.0, 3.0, 3.0) * nee_phase; // accumulate NEE contribution
#endif
        // scatter ray
        w = hg_phase_sample(w, g); // compute scattering direction in world space
        wo = scale * w; // convert scattering direction to object space for next iteration of ray marching
        // compute next distance
        ray_box_intersection(x, wo, vec3(-1.0), vec3(1.0), tMin, d); // compute intersection with volume boundary for next iteration
    }

#ifdef ENVIRONMENT
    A += W * environment(_this, w, env); // add environment contribution if ray escaped the volume
#endif
    // _output = float[](A.x, A.y, A.z);
    //A = complexity_color(bounces);
    _output = float[](A.x, A.y, A.z);
}

