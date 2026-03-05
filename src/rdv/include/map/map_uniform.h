FORWARD
{
    for (int i=0; i<OUTPUT_DIM; i++) _output[i] = random();
}

BACKWARD
{
    // No gradient computation for uniform map
}