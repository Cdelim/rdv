/* parameters
probes_map: map to sample the coordinates
field_map: map to sample the field values
rdv_indices: (deferred) optional index tensor
output_grad_tensor: gradients of the output tensor
shape: shape of the output tensor (*shape, MAP_OUTPUT_DIM)
sample_location: sampling location mode
    0: centers
    1: corners
    2: jittered cells
samples: number of samples per cell to be accumulated
*/
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_tensor { int data[INDEX_DIM]; };
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_tensor { float data[MAP_OUTPUT_DIM]; };

void rdv_generate_sample(in float cell_index[INDEX_DIM], out float probe_input[INDEX_DIM])
{
    for (int i=0; i<INDEX_DIM; i++)
        switch (parameters.sample_location[i])
        {
            case 0: // centers
                probe_input[i] = 2 * (cell_index[i] + 0.5)/parameters.shape[i] - 1;
                break;
            case 1: // corners
                probe_input[i] = 2 * cell_index[i]/(parameters.shape[i] - 1) - 1;
                break;
            case 2: // random jittered cells
                probe_input[i] = 2 * (cell_index[i] + random())/parameters.shape[i] - 1;
                break;
        }
}

MAIN(tid)
{
    // Retrieve base cell index given indices tensor or computed from tid
    float cell_index[INDEX_DIM];
    Tensor indices = load_deferred(parameters.rdv_indices);
    if (indices.data_ptr != 0) {
        rdv_input_tensor input_tensor = rdv_input_tensor(indices.data_ptr + INDEX_DIM * tid.x * 4);
        for (uint i = 0; i < INDEX_DIM; i++)
            cell_index[i] = float(input_tensor.data[INDEX_DIM - i - 1]);
    }
    else {
        uint index = tid.x;
        for (uint i = 0; i < INDEX_DIM; i++) {
            cell_index[i] = float(index % parameters.shape[i]);
            index /= parameters.shape[i];
        }
    }

    rdv_output_tensor output_grad_tensor = rdv_output_tensor(parameters.output_grad_tensor + MAP_OUTPUT_DIM * tid.x * 4);
    if (parameters.samples == 1) // no need to accumulate
    {
        // Generate sampling coordinate
        float probe_input[INDEX_DIM];
        rdv_generate_sample(cell_index, probe_input);
        // Eval the probe map
        float map_input[MAP_INPUT_DIM];
        uvec4 seed = random_seed();
        forward(parameters.probes_map, probe_input, map_input);
        // Eval the field map
    #ifdef PROBE_REQUIRES_GRAD
        float _probe_grad[MAP_INPUT_DIM];
        backward(parameters.field_map, map_input, output_grad_tensor.data, _probe_grad);
        // probe may use output
        random_seed(seed); // reset random seed before re-evaluating probe
        backward(parameters.probes_map, probe_input, map_input, _probe_grad);
    #else
        backward(parameters.field_map, map_input, output_grad_tensor.data);
    #endif
    }
    else // accumulate multiple samples
    {
        float output_grad [MAP_OUTPUT_DIM];
        for (int j=0; j<MAP_OUTPUT_DIM; j++)
            output_grad[j] = output_grad_tensor.data[j] / parameters.samples;
        for (int s=0; s < parameters.samples; s++)
        {
            // Generate sampling coordinate
            float probe_input[INDEX_DIM];
            rdv_generate_sample(cell_index, probe_input);
            // Eval the probe map
            float map_input[MAP_INPUT_DIM];
            forward(parameters.probes_map, probe_input, map_input);
            // Eval the field map
        #ifdef PROBE_REQUIRES_GRAD
            float _probe_grad[MAP_INPUT_DIM];
            backward(parameters.field_map, map_input, output_grad, _probe_grad);
            // probe may use output
            random_seed(seed); // reset random seed before re-evaluating probe
            backward(parameters.probes_map, probe_input, map_input, _probe_grad);
        #else
            backward(parameters.field_map, map_input, output_grad);
        #endif
        }
    }
}