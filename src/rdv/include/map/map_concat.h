FORWARD {
    float output_a[A_OUTPUT_DIM];
    forward(parameters.map_a, _input, output_a);
    float output_b[B_OUTPUT_DIM];
    forward(parameters.map_b, _input, output_b);
    for (int i=0; i < A_OUTPUT_DIM; i++) _output[i] = output_a[i];
    for (int i=0; i < B_OUTPUT_DIM; i++) _output[i + A_OUTPUT_DIM] = output_b[i];
}


BACKWARD {
#ifndef BW_USES_OUTPUT
    float output_a_grad[A_OUTPUT_DIM];
    for (int i=0; i < A_OUTPUT_DIM; i++) output_a_grad[i] = _output_grad[i];
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_a, _input, output_a_grad, _input_grad);
    #else
    backward(parameters.map_a, _input, output_a_grad);
    #endif
    float output_b_grad[B_OUTPUT_DIM];
    for (int i=0; i < B_OUTPUT_DIM; i++) output_b_grad[i] = _output_grad[i + A_OUTPUT_DIM];
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_b, _input, output_b_grad, _input_grad);
    #else
    backward(parameters.map_b, _input, output_b_grad);
    #endif
#else
    float output_a_grad[A_OUTPUT_DIM];
    float output_a[A_OUTPUT_DIM];
    for (int i=0; i < A_OUTPUT_DIM; i++) { output_a_grad[i] = _output_grad[i]; output_a[i] = _output[i]; }
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_a, _input, output_a, output_a_grad, _input_grad);
    #else
    backward(parameters.map_a, _input, output_a, output_a_grad);
    #endif
    float output_b_grad[B_OUTPUT_DIM];
    float output_b[B_OUTPUT_DIM];
    for (int i=0; i < B_OUTPUT_DIM; i++) { output_b_grad[i] = _output_grad[i + A_OUTPUT_DIM]; output_b[i] = _output[i + A_OUTPUT_DIM]; }
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_b, _input, output_b, output_b_grad, _input_grad);
    #else
    backward(parameters.map_b, _input, output_b, output_b_grad);
    #endif
#endif
}