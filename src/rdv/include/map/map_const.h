/*
t: deferrable_field
*/
BUFFER_DECL(t, OUTPUT_DIM)

FORWARD {
    _output = BUFFER(t) (load_tensor(parameters.t)).data;
}

BACKWARD {
    GPUPtr grad_ptr = load_tensor_grad(parameters.t);
    if (grad_ptr == 0)
    return;
    float_ptr t_grad = float_ptr(grad_ptr);
    for (int i=0; i<OUTPUT_DIM; i++)
        atomicAdd_f(t_grad, i, _output_grad[i]);
}