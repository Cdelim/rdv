FORWARD
{
    for (int i=0; i<OUTPUT_DIM / 2; i++)
    {
        vec2 v = random_normal_2();
        _output[i*2] = v.x;
        _output[i*2 + 1] = v.y;
    }
    if (OUTPUT_DIM % 2 == 1)
        _output[OUTPUT_DIM - 1] = random_normal();
    float s = 0;
    for (int i=0; i<OUTPUT_DIM; i++)
        s += _output[i] * _output[i];
    s = max(sqrt(s), 0.00000001);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] /= s;
}

BACKWARD
{
    // No gradient computation for uniform map
}