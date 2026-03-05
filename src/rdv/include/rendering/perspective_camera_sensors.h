/* Parameters
camera_poses: Nx9 array of camera poses (N origin, target, up)
fov: float
aspect_ratio: float

Maps normalized -1,1 to rays in world space
3rth value of input is which camera to use
*/

FORWARD
{
    vec3 ndc = vec3(_input[0], _input[1], 0.0);
    int camera_index = int(round((_input[2]+1.0)*0.5 * (parameters.num_cameras-1)));
    GPUPtr camera_poses = load_tensor(parameters.camera_poses);
    vec3_ptr poses_buf = vec3_ptr(camera_poses + camera_index * 9 * 4);
    vec3 o = poses_buf.data[0];
    vec3 t = poses_buf.data[1];
    vec3 n = poses_buf.data[2];

    float sx = _input[0] * parameters.aspect_ratio; //((index[2] + subsample.x) * 2 - parameters.width) * parameters.znear / parameters.width;
    float sy = _input[1]; //((index[1] + subsample.y) * 2 - parameters.height) * parameters.znear / parameters.width;
    float sz = 1.0 / tan(parameters.fov * 0.5);

    vec3 zaxis = normalize(t - o);
    vec3 xaxis = normalize(cross(n, zaxis));
    vec3 yaxis = cross(zaxis, xaxis);

    vec3 w;
    w = xaxis * sx + yaxis * sy + zaxis * sz;
    w = normalize(w);

    _output = float[6]( o.x, o.y, o.z, w.x, w.y, w.z );
}