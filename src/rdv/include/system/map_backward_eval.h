#if MAP_INPUT_DIM > 0
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_tensor { float data[MAP_INPUT_DIM]; };
#endif
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_tensor { float data[MAP_OUTPUT_DIM]; };

MAIN(tid)
{
    uint offset = tid.x * 4;
#if MAP_INPUT_DIM > 0
    rdv_input_tensor input_tensor = rdv_input_tensor(parameters.input_tensor + MAP_INPUT_DIM * offset);
#endif
    rdv_output_tensor output_grad_tensor = rdv_output_tensor(parameters.output_grad_tensor + MAP_OUTPUT_DIM * offset);
#if MAP_INPUT_DIM == 0
    backward(parameters.map, output_grad_tensor.data);
#else
#ifndef INPUT_REQUIRES_GRAD
    backward(parameters.map, input_tensor.data, output_grad_tensor.data);
#else
    rdv_input_tensor input_grad_tensor = rdv_input_tensor(parameters.input_grad_tensor + MAP_INPUT_DIM * offset);
    for (int i=0; i<MAP_INPUT_DIM; i++) input_grad_tensor.data[i] = 0.0;
    backward(parameters.map, input_tensor.data, output_grad_tensor.data, input_grad_tensor.data);
#endif
#endif
}