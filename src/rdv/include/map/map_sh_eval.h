FORWARD {
    vec3 w = vec3(_input[0], _input[1], _input[2]);
    eval_sh(w, _output);
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    vec3 w = vec3(_input[0], _input[1], _input[2]);
    vec3 dw;
    eval_sh_grad(w, _output_grad, dw);
    _input_grad = float[] (dw.x, dw.y, dw.z);
#endif
}