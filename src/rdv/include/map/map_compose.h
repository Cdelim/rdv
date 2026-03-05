FORWARD {
    float intermediate [INTERMEDIATE_DIM];
    forward(parameters.inner, _input, intermediate);
    forward(parameters.outer, intermediate, _output);
}

BACKWARD {
    float intermediate [INTERMEDIATE_DIM];
#ifdef STOCHASTIC
    uvec4 seed = random_seed(); // gets current seed before forward a
#endif
    forward(parameters.inner, _input, intermediate);
    #ifdef INNER_REQUIRES_GRAD
    float intermediate_grad [INTERMEDIATE_DIM];
    for (int i=0; i<INTERMEDIATE_DIM; i++) intermediate_grad[i] = 0.0;
    #endif
    #ifndef BW_USES_OUTPUT
    backward(parameters.outer, intermediate, _output_grad
    #ifdef INNER_REQUIRES_GRAD
    , intermediate_grad
    #endif
    );
    #else
    backward(parameters.outer, intermediate, _output, _output_grad
    #ifdef INNER_REQUIRES_GRAD
    , intermediate_grad
    #endif
    );
    #endif
#ifdef STOCHASTIC
    seed = random_seed(seed); // sets again the seed before forward a and saves end seed
#endif
    // backward(parameters.inner, _input, intermediate, intermediate_grad, _input_grad);
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.inner, _input, intermediate_grad, _input_grad);
    #else
    #ifdef INNER_REQUIRES_GRAD
    backward(parameters.inner, _input, intermediate_grad);
    #endif
    #endif
#ifdef STOCHASTIC
    random_seed(seed); // restores end seed
#endif
}