float transmittance_rt(MAP_DECL, vec3 x, vec3 w, float d)
{
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
