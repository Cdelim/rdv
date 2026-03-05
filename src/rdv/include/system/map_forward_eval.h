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
    rdv_output_tensor output_tensor = rdv_output_tensor(parameters.output_tensor + MAP_OUTPUT_DIM * offset);
#if MAP_INPUT_DIM > 0
    forward(parameters.map, input_tensor.data, output_tensor.data);
#else
    forward(parameters.map, output_tensor.data);
#endif
}