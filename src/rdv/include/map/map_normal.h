FORWARD
{
    int i=0;
    while (i < OUTPUT_DIM)
    {
        vec2 r = random_normal_2();
        _output[i++] = r.x;
        if (i < OUTPUT_DIM)
            _output[i++] = r.y;
    }
}

BACKWARD
{
    // No gradient computation for normal map
}