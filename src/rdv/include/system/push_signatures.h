#define FORWARD void forward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], out float _output[ARRAY_SIZE(OUTPUT_DIM)])
#ifdef INPUT_REQUIRES_GRAD
    #ifdef BW_USES_OUTPUT
        #define BACKWARD void backward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output[ARRAY_SIZE(OUTPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)], inout float _input_grad[ARRAY_SIZE(INPUT_DIM)])
    #else
        #define BACKWARD void backward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)], inout float _input_grad[ARRAY_SIZE(INPUT_DIM)])
    #endif
#else
    #ifndef PARAMS_REQUIRES_GRAD // If parameters do not require grad (neither input), create discarded version
        #ifdef BW_USES_OUTPUT
            #define BACKWARD void backward_discarded(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output[ARRAY_SIZE(OUTPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)])
        #else
            #define BACKWARD void backward_discarded(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)])
        #endif
    #else
        #ifdef BW_USES_OUTPUT
            #define BACKWARD void backward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output[ARRAY_SIZE(OUTPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)])
        #else
            #define BACKWARD void backward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)])
        #endif
    #endif
#endif

// Create empty definitions to be used in macros
FORWARD;
BACKWARD;

#if !defined(INPUT_REQUIRES_GRAD) && !defined(PARAMS_REQUIRES_GRAD)
    #ifdef BW_USES_OUTPUT
        void backward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output[ARRAY_SIZE(OUTPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)]){}
    #else
        void backward(MAP_DECL, in float _input[ARRAY_SIZE(INPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)]){}
    #endif
#endif

#if INPUT_DIM == 0
void forward(MAP_DECL, out float _output[ARRAY_SIZE(OUTPUT_DIM)]) {
    // No input, just call forward with dummy input
    float dummy_input[1];
    forward(_this, dummy_input, _output);
}
#ifdef BW_USES_OUTPUT
void backward(MAP_DECL, in float _output[ARRAY_SIZE(OUTPUT_DIM)], in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)]) {
    // No input, just call backward with dummy input
    float dummy_input[1];
    backward(_this, dummy_input, _output, _output_grad);
}
#else
void backward(MAP_DECL, in float _output_grad[ARRAY_SIZE(OUTPUT_DIM)]) {
    // No input, just call backward with dummy input
    float dummy_input[1];
    backward(_this, dummy_input, _output_grad);
}
#endif
#endif


#define COMPONENT_WISE_MAP \
float forward(map_object, float x); \
void backward(map_object, float x, float y, float y_grad, inout float x_grad); \
FORWARD { /*[[unroll]]*/ for (int i=0; i<INPUT_DIM; i++) _output[i] = forward(rdv_map, _input[i]); } \
BACKWARD { /*[[unroll]]*/ for (int i=0; i<INPUT_DIM; i++) \
#ifdef INPUT_REQUIRES_GRAD \
#ifndef BW_USES_OUTPUT \
backward(rdv_map, _input[i], 0, _output_grad[i], _input_grad[i]); \
#else \
backward(rdv_map, _input[i], _output[i], _output_grad[i], _input_grad[i]); \
#endif \
#else \
float x_grad; \
#ifndef BW_USES_OUTPUT \
backward(rdv_map, _input[i], 0, _output_grad[i], x_grad); \
#else \
backward(rdv_map, _input[i], _output[i], _output_grad[i], x_grad); \
#endif \
#endif \
}

