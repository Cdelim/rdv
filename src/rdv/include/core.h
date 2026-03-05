#ifndef CORE_H
#define CORE_H

#extension GL_EXT_debug_printf : enable
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2: require
#extension GL_ARB_gpu_shader_int64 : require
#ifdef SUPPORTED_RAY_QUERY
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing : require
#endif
#ifdef SUPPORTED_FLOAT_ATOM_ADD
#extension GL_EXT_shader_atomic_float : require
#endif
#extension GL_EXT_control_flow_attributes : require

/*
==================================
              POINTERS
==================================
*/
#define GPUPtr uint64_t
#define POINTER(type_name, align) layout(buffer_reference, scalar, buffer_reference_align=align) buffer type_name##_ptr { type_name data[]; };
POINTER(float, 4)
POINTER(int, 4)
POINTER(uint, 4)
POINTER(GPUPtr, 4)
POINTER(vec4, 4)
POINTER(vec3, 4)
POINTER(vec2, 4)
POINTER(mat2, 4)
POINTER(mat3, 4)
POINTER(mat4, 4)
POINTER(mat2x3, 4)
POINTER(mat2x4, 4)
POINTER(mat3x2, 4)
POINTER(mat3x4, 4)
POINTER(mat4x2, 4)
POINTER(mat4x3, 4)
POINTER(ivec4, 4)
POINTER(ivec3, 4)
POINTER(ivec2, 4)
POINTER(uvec4, 4)
POINTER(uvec3, 4)
POINTER(uvec2, 4)

/*
==================================
              Debug
==================================
*/
#define PRINT debugPrintfEXT
#define ASSERT(cond, message) if (!(cond)) debugPrintfEXT(message);


vec3 complexity_color(int value) {
    vec3 colors [8] = vec3[] (
        vec3(0.0, 0.0, 0.2),
        vec3(0.0, 0.2, 0.4),
        vec3(0.0, 0.4, 0.6),
        vec3(0.0, 0.8, 0.2),
        vec3(0.2, 0.8, 0.0),
        vec3(0.4, 0.6, 0.0),
        vec3(0.8, 0.2, 0.0),
        vec3(1.0, 0.0, 0.0)
    );
    float alpha = log(float(value) + 1.0) / log(10000.0); // Assuming max value is around 1000 for normalization
    alpha = clamp(alpha * 8.0, 0.0, 7.0); // Scale alpha to [0, 8) and clamp to valid index
    return mix(colors[int(floor(alpha))], colors[int(ceil(alpha))], fract(alpha));
}

/*
==================================
              MATH
==================================
*/
#define pi 3.1415926535897932384626433832795
#define piOverTwo 1.5707963267948966192313216916398
#define inverseOfPi 0.31830988618379067153776752674503
#define inverseOfTwoPi 0.15915494309189533576888376337251
#define two_pi 6.283185307179586476925286766559
#define POSINF uintBitsToFloat(0x7f800000u)
#define NEGINF uintBitsToFloat(0xff800000u)


bool intersect_ray_box(vec3 x, vec3 w, vec3 b_min, vec3 b_max, out float tMin, out float tMax)
{
    // un-parallelize w
    vec3 C_Min = (b_min - x)/w;
    vec3 C_Max = (b_max - x)/w;
	tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
	tMin = max(0.0, tMin);
	tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
	if (tMax <= tMin || tMax <= 0) {
		return false;
	}
	return true;
}

void ray_box_intersection(vec3 x, vec3 w, vec3 b_min, vec3 b_max, out float tMin, out float tMax)
{
    vec3 C_Min = (b_min - x)/w;
    vec3 C_Max = (b_max - x)/w;
    vec3 c = min(C_Min, C_Max);
	tMin = max(max(c.x, c.y), c.z);
	c = max(C_Min, C_Max);
	tMax = min(min(c.x, c.y), c.z);
}

void ray_box_intersection(vec3 x, vec3 w, out float tMin, out float tMax)
{
    vec3 inv_dist = 1.0 / w;
    vec3 xc = -x * inv_dist;
    vec3 s = abs(inv_dist);
    vec3 c = xc-s;
    tMin = max(max(c.x, c.y), c.z);
    c = xc+s;
    tMax = min(min(c.x, c.y),  c.z);
}

void segment_box_intersection(vec3 x0, vec3 x1, vec3 bmin, vec3 bmax, out float tMin, out float tMax)
{
    vec3 inv_dist = 1/(x1 - x0);
    vec3 cmin = (bmin - x0)*inv_dist;
    vec3 cmax = (bmax - x0)*inv_dist;
    vec3 m = min(cmin, cmax);
    vec3 M = max(cmin, cmax);
    tMin = max(0.0, max(max(m.x, m.y), m.z));
    tMax = min(1.0, min(min(M.x, M.y), M.z));
}

/*
Segment intersection against axis-aligned box [-1,1]^3
*/
void segment_box_intersection(vec3 x0, vec3 x1, out float tMin, out float tMax)
{
    vec3 inv_dist = 1.0 / (x1 - x0);
    vec3 x0_div_dist = x0 * inv_dist;
    vec3 c = abs(inv_dist) - x0_div_dist;
    tMin = max(0.0, max(max(-c.x, -c.y), -c.z));
    tMax = min(1.0, min(min(c.x, c.y), c.z));
}

void ray_sphere_intersection(vec3 x, vec3 w, vec3 center, float radius, out float tMin, out float tMax)
{
    vec3 oc = x - center;
    float a = dot(w, w);
    float b = 2.0 * dot(oc, w);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        tMin = POSINF;
        tMax = NEGINF;
    } else {
        float sqrt_disc = sqrt(discriminant);
        tMin = (-b - sqrt_disc) / (2.0 * a);
        tMax = (-b + sqrt_disc) / (2.0 * a);
    }
}

// https://github.com/google/spherical-harmonics


#define _C0 0.28209479177387814
#define _C1 0.4886025119029199
#define _C20 1.0925484305920792
#define _C21 -1.0925484305920792
#define _C22 0.31539156525252005
#define _C23 -1.0925484305920792
#define _C24 0.5462742152960396

const float _C3[7] = float[7](
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

//
//const float C4[9] = float[9](
//     2.5033429417967046,
//     -1.7701307697799304,
//     0.9461746957575601,
//     -0.6690465435572892,
//     0.10578554691520431,
//     -0.6690465435572892,
//     0.47308734787878004,
//     -1.7701307697799304,
//     0.6258357354491761
// );


void eval_sh(vec3 w, out float coef[1])
{
    coef[0] = _C0;
}

void eval_sh_grad(vec3 w, float coef_grad[1], out vec3 dw)
{
    dw = vec3(0.0);
}

void eval_sh(vec3 w, out float coef[4])
{
    coef[0] = _C0;
    float x = w.x, y = w.y, z = w.z;
    coef[1] = -_C1 * y;
    coef[2] = _C1 * z;
    coef[3] = -_C1 * x;
}

void eval_sh_grad(vec3 w, float coef_grad[4], out vec3 dw)
{
    dw = vec3(0.0);
    dw.x -= coef_grad[3] * _C1;
    dw.y -= coef_grad[1] * _C1;
    dw.z += coef_grad[2] * _C1;
}


void eval_sh(vec3 w, out float coef[9])
{
    coef[0] = _C0;
    float x = w.x, y = w.y, z = w.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    coef[4] = _C20 * xy;
    coef[5] = _C21 * yz;
    coef[6] = _C22 * (2.0 * zz - xx - yy);
    coef[7] = _C23 * xz;
    coef[8] = _C24 * (xx - yy);
    coef[1] = -_C1 * y;
    coef[2] = _C1 * z;
    coef[3] = -_C1 * x;
}

void eval_sh_grad(vec3 w, float coef_grad[9], out vec3 dw)
{
    float x = w.x, y = w.y, z = w.z;
    dw = vec3(0.0);
    dw.x -= coef_grad[3] * _C1;
    dw.y -= coef_grad[1] * _C1;
    dw.z += coef_grad[2] * _C1;
    // coef[4] = _C20 * xy;
    dw.x += coef_grad[4] * _C20 * y;
    dw.y += coef_grad[4] * _C20 * x;
    // coef[5] = _C21 * yz;
    dw.y += coef_grad[5] * _C21 * z;
    dw.z += coef_grad[5] * _C21 * y;
    // coef[6] = _C22 * (2.0 * zz - xx - yy);
    dw.x -= coef_grad[6] * 2 * _C22;
    dw.y -= coef_grad[6] * 2 * _C22;
    dw.z += coef_grad[6] * 4 * _C22;
    // coef[7] = _C23 * xz;
    dw.x += coef_grad[7] * _C23 * z;
    dw.z += coef_grad[7] * _C23 * x;
    // coef[8] = _C24 * (xx - yy);
    dw.x += coef_grad[8] * _C24 * 2;
    dw.y -= coef_grad[8] * _C24 * 2;
}

void eval_sh(vec3 w, out float coef[16])
{
    coef[0] = _C0;
    float x = w.x, y = w.y, z = w.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    coef[4] = _C20 * xy;
    coef[5] = _C21 * yz;
    coef[6] = _C22 * (2.0 * zz - xx - yy);
    coef[7] = _C23 * xz;
    coef[8] = _C24 * (xx - yy);
    coef[1] = -_C1 * y;
    coef[2] = _C1 * z;
    coef[3] = -_C1 * x;
    coef[9] = _C3[0] * y * (3 * xx - yy);
    coef[10] = _C3[1] * xy * z;
    coef[11] = _C3[2] * y * (4 * zz - xx - yy);
    coef[12] = _C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
    coef[13] = _C3[4] * x * (4 * zz - xx - yy);
    coef[14] = _C3[5] * z * (xx - yy);
    coef[15] = _C3[6] * x * (xx - 3 * yy);
}
// TODO: Implement eval_sh_grad for 16

mat4 look_at_LH(vec3 camera, vec3 target, vec3 upVector)
{
	vec3 zaxis = normalize(target - camera);
	vec3 xaxis = normalize(cross(upVector, zaxis));
	vec3 yaxis = cross(zaxis, xaxis);

	return mat4(
		xaxis.x, yaxis.x, zaxis.x, 0,
		xaxis.y, yaxis.y, zaxis.y, 0,
		xaxis.z, yaxis.z, zaxis.z, 0,
		-dot(xaxis, camera), -dot(yaxis, camera), -dot(zaxis, camera), 1);
}

mat4 perspective_fov_LH(float fieldOfView, float aspectRatio, float znearPlane, float zfarPlane)
{
	float h = 1.0 / tan(fieldOfView / 2.0);
	float w = h / aspectRatio;

	return mat4(
		w, 0, 0, 0,
		0, h, 0, 0,
		0, 0, zfarPlane / (zfarPlane - znearPlane), 1,
		0, 0, -znearPlane * zfarPlane / (zfarPlane - znearPlane), 0);
}

/*
Converts angle from azimuth (angles.x) [-pi...pi] and polar (angles.y) [-pi/2...pi/2] to a direction.
angle 0,0 corresponds to forward direction in z.
*/
vec3 usc2dir(vec2 angles)
{
    float y = sin(angles.y);
    float r = cos(angles.y);
    float x = sin(angles.x) * r;
    float z = cos(angles.x) * r;
    return vec3(x, y, z);
}

vec2 dir2usc(vec3 w) {
    w.x += 0.0000001 * int(w.x == 0.0 && w.z == 0.0);
    float beta = asin(clamp(w.y, -1.0, 1.0));
    float alpha = atan(w.x, w.z);
    return vec2(alpha, beta);
}

vec2 dir2xr(vec3 w)
{
    vec2 angles = dir2usc(w);
    return angles * vec2(inverseOfPi, 2*inverseOfPi);
}

vec3 xr2dir(vec2 c)
{
    vec2 angles = c * vec2(pi, piOverTwo);
    return usc2dir(angles);
}

// Adapted from:
// https://gamedev.stackexchange.com/questions/169508/octahedral-impostors-octahedral-mapping
vec2 dir2oct(vec3 w)
{
    vec3 octant = sign(w);
    // Scale the vector so |x| + |y| + |z| = 1 (surface of octahedron).
    float sum = dot(w, octant);
    vec3 octahedron = w / sum;

    // "Untuck" the corners using the same reflection across the diagonal as before.
    // (A reflection is its own inverse transformation).
    if(octahedron.z < 0) {
        vec3 absolute = abs(octahedron);
        octahedron.xy = octant.xy
                      * vec2(1.0 - absolute.y, 1.0 - absolute.x);
    }
    return octahedron.xy;
}

vec3 oct2dir(vec2 c)
{
    // Unpack the 0...1 range to the -1...1 unit square.
    vec3 w = vec3(c, 0);

    // "Lift" the middle of the square to +1 z, and let it fall off linearly
    // to z = 0 along the Manhattan metric diamond (absolute.x + absolute.y == 1),
    // and to z = -1 at the corners where position.x and .y are both = +-1.
    vec2 absolute = abs(w.xy);
    w.z = 1.0 - absolute.x - absolute.y;

    // "Tuck in" the corners by reflecting the xy position along the line y = 1 - x
    // (in quadrant 1), and its mirrored image in the other quadrants.
    if(w.z < 0)
        w.xy = sign(w.xy)
                    * vec2(1.0 - absolute.y, 1.0 - absolute.x);

    return normalize(w);
}

void create_ortho_basis(vec3 N, out vec3 T, out vec3 B)
{
    float sign = N.z >= 0 ? 1 : -1;
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = vec3(b, sign + N.y * N.y * a, -N.y);
}

// MORTON CODE FUNCTIONS
// Adapted from https://gist.github.com/wontonst/8696dcfb643121c864dec7c0d6ad26c5
int part1by1(int n){
    n &= 0x0000ffff;
    n = (n | (n << 8)) & 0x00FF00FF;
    n = (n | (n << 4)) & 0x0F0F0F0F;
    n = (n | (n << 2)) & 0x33333333;
    n = (n | (n << 1)) & 0x55555555;
    return n;
}

int unpart1by1(int n){
    n &= 0x55555555; // base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = (n ^ (n >> 1)) & 0x33333333; // base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n ^ (n >> 2)) & 0x0f0f0f0f; // base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n ^ (n >> 4)) & 0x00ff00ff; // base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n ^ (n >> 8)) & 0x0000ffff; // base10: 65535,      binary: 1111111111111111,                 len: 16
    return n;
}

void morton2pixel(int index, out int px, out int py)
{
    px = unpart1by1(index);
    py = unpart1by1(index >> 1);
}

ivec2 morton2pixel(int index)
{
    return ivec2(unpart1by1(index), unpart1by1(index >> 1));
}

void pixel2morton(int px, int py, out int index)
{
    index = part1by1(px) | (part1by1(py) << 1);
}

int pixel2morton(ivec2 px)
{
    return part1by1(px.x) | (part1by1(px.y) << 1);
}

/*
==================================
              RANDOMS
==================================
*/

#ifdef RDV_STOCHASTIC_COMPUTE

/*
Random number generator is only compiled if RDV_STOCHASTIC_COMPUTE is defined.
If a map requires randomness, the whole kernel is compiled with this flag.
Specify map randomness requirements in the map __extension_info__ structure with stochastic=True.
*/

// adapted from NVidia
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
uint TausStep(uint z, int S1, int S2, int S3, uint M) { uint b = (((z << S1) ^ z) >> S2); return ((z & M) << S3) ^ b; }
uint LCGStep(uint z, uint A, uint C) { return A * z + C; }

uvec4 rdv_rng_state;

void random_step(inout uvec4 rng_state)
{
	rng_state.x = TausStep(rng_state.x, 13, 19, 12, 4294967294U);
	rng_state.y = TausStep(rng_state.y, 2, 25, 4, 4294967288U);
	rng_state.z = TausStep(rng_state.z, 3, 11, 17, 4294967280U);
	rng_state.w = LCGStep(rng_state.w, 1664525, 1013904223U);
}

void random_step(){
    random_step(rdv_rng_state);
}

uvec4 random_seed()
{
    return rdv_rng_state;
}

float random(inout uvec4 rng_state)
{
    random_step(rng_state);
	uint v = rng_state.x ^ rng_state.y ^ rng_state.z ^ rng_state.w;
	// more robust computation
	return uintBitsToFloat(v & 0x007FFFFF | 0x3F800000) - 1;
	// this has an error and produces randoms = 1
    // f = 2.3283064364387e-10 * uint(rdv_rng_state.x ^ rdv_rng_state.y ^ rdv_rng_state.z ^ rdv_rng_state.w); // THERE WAS AN ERROR HERE!
}

float random()
{
    return random(rdv_rng_state);
}

uvec4 random_spawn(uvec4 rng_state, int index)
{
    rng_state = rng_state ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (rng_state.x + 13 * rng_state.y));
    // uvec4 state = rng_state ^ uvec4(0x23F1 * index + index*(~index),0x3137 % index, index ^ index, index + 129);
    random_step(rng_state);
    return rng_state;
}

uvec4 random_seed(uvec4 rng_state)
{
    uvec4 old_state = rdv_rng_state;
    rdv_rng_state = rng_state;
    return old_state;
}

uvec4 random_branch()
{
    uvec4 old_rng = rdv_rng_state;
    rdv_rng_state = floatBitsToUint(vec4(random(), random(), random(), random())) + 129;
    random_step();
    return old_rng;
}

vec3 random_direction(vec3 D) {
	float r1 = random();
	float r2 = random() * 2 - 1;
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	create_ortho_basis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction()
{
    return random_direction(vec3(0.0, 1.0, 0.0));
}

vec3 random_direction(vec3 D, float fov) {
	float r1 = random();
	float r2 = 1 - random() * (1 - cos(fov));
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	create_ortho_basis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction(float alpha0, float alpha1, float beta0, float beta1) {
	float r1 = random() * (alpha1 - alpha0) + alpha0;
	float r2 = sin(beta0) + random() * (sin(beta1) - sin(beta0));
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0 = vec3(1, 0, 0);
	vec3 t1 = vec3(0, 0, 1);
	vec3 D = vec3(0, 1, 0);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction_HS(vec3 D) {
	float r1 = random();
	float r2 = random();
	float sqrR2 = r2 * r2;
	float two_pi_by_r1 = two_pi * r1;
	float sqrt_of_one_minus_sqrR2 = sqrt(max(0, 1.0 - sqrR2));
	float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
	float z = r2;
	vec3 t0, t1;
	create_ortho_basis(D, t0, t1);
	return t0 * x + t1 * y + D * z;
}

vec3 random_direction_HS_cosine_weighted(vec3 N, out float NdotD)
{
	vec3 t0, t1;
	create_ortho_basis(N, t0, t1);

	while (true) {
		float x = random() * 2 - 1;
		float y = random() * 2 - 1;
		float d2 = x * x + y * y;
		if (d2 > 0.001 && d2 < 1)
		{
			float z = sqrt(1 - d2);
			NdotD = z;
			return t0 * x + t1 * y + N * z;
		}
	}
	return vec3(0,0,0);
}

vec3 random_direction_HS_cosine_weighted()
{
	while (true) {
		float x = random() * 2 - 1;
		float y = random() * 2 - 1;
		float d2 = x * x + y * y;
		if (d2 > 0.001 && d2 < 1)
		{
			float z = sqrt(1 - d2);
			return vec3(x, y, z);
		}
	}
	return vec3(0,0,0);
}



vec2 rdv_BM() {
	float u1 = 1.0 - random(); //uniform(0,1] random doubles
	float u2 = 1.0 - random();
	float r = sqrt(-2.0 * log(max(0.0000000001, u1)));
	float t = 2.0 * pi * u2;
	return r * vec2(cos(t), sin(t));
}

float random_normal() {
	return rdv_BM().x;
}

vec2 random_normal_2()
{
    return rdv_BM();
}

vec3 random_normal_3()
{
    return vec3(rdv_BM(), rdv_BM().x);
}

vec4 random_normal_4()
{
    return vec4(rdv_BM(), rdv_BM());
}

float random_normal(float mu, float sd)
{
    return sd * random_normal() + mu;
}

vec2 random_normal(vec2 mu, vec2 sd)
{
    return sd * random_normal_2() + mu;
}

vec3 random_normal(vec3 mu, vec3 sd)
{
    return sd * random_normal_3() + mu;
}

vec4 random_normal(vec4 mu, vec4 sd)
{
    return sd * random_normal_4() + mu;
}

#endif

/*
==================================
              SUPPORT
==================================
*/

float atomicAdd_f(float_ptr buf, int index, float value)
{
    #ifdef SUPPORTED_FLOAT_ATOM_ADD
    return atomicAdd(buf.data[index], value);
    #else
    uint_ptr buf_as_uint = uint_ptr(uint64_t(buf));
    uint old = buf_as_uint.data[index];
    uint assumed;
    do {
        assumed = old;
        old = atomicCompSwap(buf_as_uint.data[index], assumed, floatBitsToUint(value + uintBitsToFloat(assumed)));
    } while(assumed != old);
    return uintBitsToFloat(old);
    #endif
}

/*
==================================
              TORCH
==================================
*/


layout(set = 0, scalar, binding = 1) uniform DeferredDataBuffer {
    GPUPtr data[1024];
    GPUPtr grad_data[1024];
} rdv_deferred_data_buffer;


GPUPtr load_deferred_tensor(in GPUPtr t) {
    uvec2 data = unpackUint2x32(t);
    return rdv_deferred_data_buffer.data[data.x >> 1] + data.y; // data.y is the offset, data.x >> 1 is the index in the deferred buffer
}


GPUPtr load_tensor(in GPUPtr t) {
#ifdef RDV_HAS_DEFERRED
    if (t % 2 == 0)  // even means it's a direct pointer, odd means it's a deferred tensor (pointer to the deferred buffer)
        return t;
    else
    {
        uvec2 data = unpackUint2x32(t);
        return rdv_deferred_data_buffer.data[data.x >> 1] + data.y; // data.y is the offset, data.x >> 1 is the index in the deferred buffer
    }
#else
    return t;
#endif
}

GPUPtr load_tensor_grad(in GPUPtr t) {
#ifdef RDV_HAS_DEFERRED
    if (t % 2 == 0)  // even means it's a direct pointer, odd means it's a deferred tensor (pointer to the deferred buffer)
        return t;
    else
    {
        uvec2 data = unpackUint2x32(t);
        return rdv_deferred_data_buffer.grad_data[data.x >> 1] + data.y; // data.y is the offset, data.x >> 1 is the index in the deferred buffer
    }
#else
    return 0;
#endif
}

//struct Tensor { GPUPtr data_ptr; GPUPtr grad_ptr; uint shape[5]; };
//
//Tensor load_deferred(in DeferrableField p) {
//    if (p.deferred_index < 0) {
//        Tensor t;
//        t.shape = p.shape;
//        t.data_ptr = p.data;
//        t.grad_ptr = 0;
//        return t;
//    }
//    else {
//        Tensor t = Tensor(0, 0, uint[5](0,0,0,0,0));
//        rdv_DeferredParameterInfo dinfo = rdv_deferred_buffer.data[p.deferred_index];
//        rdv_NamedTensorInfo ninfo = rdv_named_buffer.data[dinfo.name_id];
//        if (ninfo.data == 0)  // null tensor
//            return t;
////        PRINT("loaded not null deferred tensor index %d", p.deferred_index);
//        uint map_tensor_size = 1;
//        int b = dinfo.map_dim + dinfo.number_of_indices - ninfo.dim;  // values to pad
//        for(int i=0; i < dinfo.map_dim; i++)
//        {
//            uint d = i < b ? 1: ninfo.shape[dinfo.number_of_indices + i - b];
//            map_tensor_size *= d;
//            t.shape[i] = d;
//        }
//        uint acc_batch_size = map_tensor_size;
//        uint offset = 0;
//        for(int i=dinfo.number_of_indices - 1; i >= 0; i++)
//        {
//            offset += acc_batch_size * dinfo.indices[i];
//            acc_batch_size *= ninfo.shape[i];
//        }
//        t.data_ptr = ninfo.data + offset;
//        t.grad_ptr = ninfo.grad_data == 0 ? 0 : ninfo.grad_data + offset;
//        return t;
//    }
//}

//#define DECLARE_INDEXING(dim) uint index_offset(uint shape[5], int index[dim]) {
//    uint offset = 0; uint acc = 1;
//    for (int i = dim - 1; i >= 0; i--) {
//        offset += index[i] * acc; acc *= shape[i]; }
//    return offset; }
//
//DECLARE_INDEXING(1)
//
//DECLARE_INDEXING(2)
//
//DECLARE_INDEXING(3)
//
//DECLARE_INDEXING(4)
//
//DECLARE_INDEXING(5)
//
//float_ptr tensor_at(in Tensor t, ivec3 index) {
//    return float_ptr(t.data_ptr + index_offset(t.shape, int[](index.z, index.y, index.x)) * 4);
//}

float_ptr tensor_at(GPUPtr base_ptr, int dim, int shape[3], ivec3 p)
{
    int offset = ((p.z * shape[1] + p.y) * shape[2] + p.x) * 4 * dim;
    return float_ptr(base_ptr + offset);
}

/*
==================================
              Geometry
==================================
*/

layout(buffer_reference, scalar, buffer_reference_align=4) buffer MeshInfo {
    GPUPtr positions;
    GPUPtr normals;
    GPUPtr coordinates;
    GPUPtr tangents;
    GPUPtr binormals;
    GPUPtr indices;
};

layout(buffer_reference, scalar, buffer_reference_align=4) buffer RaycastableInfo {
    GPUPtr callable_map;
    GPUPtr mesh_info;
};

struct Surfel
{
    vec3 P; // Position at the surface
    vec3 N; // Shading normal at the surface (might differ from real normal G)
    vec3 G; // Gradient vector at the surface pointing 'outside'
    vec2 C; // Coordinates to parameterize the surface
    vec3 T; // Tangent vector of the parameterization
    vec3 B; // Binormal vector of the parameterization
};

bool hit2surfel (vec3 x, vec3 w, in float[16] a, out float t, out int patch_index, out Surfel surfel)
{
    patch_index = floatBitsToInt(a[15]);
    if (patch_index == -1)
        return false;
    t = a[0];
    surfel.P = w * t + x;
    surfel.N = vec3(a[1], a[2], a[3]);
    surfel.G = vec3(a[4], a[5], a[6]);
    surfel.C = vec2(a[7], a[8]);
    surfel.T = vec3(a[9], a[10], a[11]);
    surfel.B = vec3(a[12], a[13], a[14]);
    return true;
}

void surfel2array(float t, int patch_index, Surfel surfel, out float[16] a)
{
    a = float[16](
        t,
        surfel.N.x,surfel.N.y,surfel.N.z,
        surfel.G.x,surfel.G.y,surfel.G.z,
        surfel.C.x,surfel.C.y,
        surfel.T.x,surfel.T.y,surfel.T.z,
        surfel.B.x,surfel.B.y,surfel.B.z,
        intBitsToFloat(patch_index)
    );
}

void noHit2array(out float[16] a)
{
    a = float[16](
        0,
        0, 0, 0,
        0, 0, 0,
        0, 0,
        0, 0, 0,
        0, 0, 0,
        intBitsToFloat(-1)
    );
}

vec3 transform_position(vec3 P, mat4 T)
{
    return (T * vec4(P, 1)).xyz;
}

vec3 transform_position(vec3 P, mat4x3 T)
{
    return (T * vec4(P, 1));
}

vec3 transform_normal(vec3 N, mat4 T)
{
    mat3 T_N = mat3(T);
    return normalize(transpose(inverse(T_N)) * N);
}

vec3 transform_normal(vec3 N, mat4x3 T)
{
    mat3 T_N = mat3(T);
    return normalize(transpose(inverse(T_N)) * N);
}

vec3 transform_direction(vec3 D, mat4 T)
{
    return (T * vec4(D, 0)).xyz;
}

vec3 transform_direction(vec3 D, mat4x3 T)
{
    return (T * vec4(D, 0));
}

Surfel transform(Surfel surfel, mat4 T)
{
    Surfel result;
    result.P = transform_position(surfel.P, T);
    result.N = transform_normal(surfel.N, T);
    result.G = transform_normal(surfel.G, T);
    result.C = surfel.C;
    result.T = transform_direction(surfel.T, T);
    result.B = transform_direction(surfel.B, T);
    return result;
}

Surfel transform(Surfel surfel, mat4x3 T)
{
    Surfel result;
    result.P = transform_position(surfel.P, T);
    result.N = transform_normal(surfel.N, T);
    result.G = transform_normal(surfel.G, T);
    result.C = surfel.C;
    result.T = transform_direction(surfel.T, T);
    result.B = transform_direction(surfel.B, T);
    return result;
}

mat4x3 inverse_transform(mat4x3 T)
{
    mat4 M = mat4(T);
    M[3][3] = 1.0;
    M = inverse(M);
    return mat4x3(M);
}

void transform_ray_to_object(inout vec3 x, inout vec3 w, mat4x3 T)
{
    mat3 L = inverse(mat3(T[0].xyz, T[1].xyz, T[2].xyz));
    vec3 t = T[3].xyz;
    x = L * (x - t);
    w = L * w;
}

void transform_ray_to_world(inout vec3 x, inout vec3 w, mat4x3 T)
{
    mat3 L = mat3(T[0].xyz, T[1].xyz, T[2].xyz);
    vec3 t = T[3].xyz;
    x = L * x + t;
    w = L * w;
}

void transform_normal_to_world(inout vec3 N, mat4x3 T)
{
    mat3 L = mat3(T[0].xyz, T[1].xyz, T[2].xyz);
    N = normalize(transpose(inverse(L)) * N);
}

Surfel sample_surfel(in MeshInfo mesh, int index, vec2 baricentrics)
{
    vec3 alphas = vec3(1 - baricentrics.x - baricentrics.y, baricentrics.x, baricentrics.y);

    int idx0, idx1, idx2;

    int i = index * 3;
    if (mesh.indices != 0)
    {
        int_ptr idxs = int_ptr(mesh.indices);
        idx0 = idxs.data[i++];
        idx1 = idxs.data[i++];
        idx2 = idxs.data[i];
    }
    else
    {
        idx0 = i++;
        idx1 = i++;
        idx2 = i;
    }

    vec3_ptr pos = vec3_ptr(mesh.positions);
    vec3 P = pos.data[idx0] * alphas.x + pos.data[idx1] * alphas.y + pos.data[idx2] * alphas.z;
    vec3 Nface = normalize(cross(pos.data[idx1] - pos.data[idx0], pos.data[idx2] - pos.data[idx0]));
    vec3_ptr nor = vec3_ptr(mesh.normals);
    vec3 N = mesh.normals == 0 ? Nface : normalize(nor.data[idx0] * alphas.x + nor.data[idx1] * alphas.y + nor.data[idx2] * alphas.z);
    vec2_ptr coordinates = vec2_ptr(mesh.coordinates);
    vec2 C = mesh.coordinates == 0 ? vec2(0.0) : coordinates.data[idx0] * alphas.x + coordinates.data[idx1] * alphas.y + coordinates.data[idx2] * alphas.z;
    vec3_ptr tang = vec3_ptr(mesh.tangents);
    vec3 T = mesh.tangents == 0 ? vec3(0.0) : tang.data[idx0] * alphas.x + tang.data[idx1] * alphas.y + tang.data[idx2] * alphas.z;
    vec3_ptr bin = vec3_ptr(mesh.binormals);
    vec3 B = mesh.binormals == 0 ? vec3(0.0) : bin.data[idx0] * alphas.x + bin.data[idx1] * alphas.y + bin.data[idx2] * alphas.z;

    return Surfel(P, N, Nface, C, T, B);
}

#ifdef RDV_STOCHASTIC_COMPUTE

vec3 hg_phase_sample(vec3 w_in, float g) {
	float phi = random() * 2 * pi;
    float xi = random();
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    float one_over_2g = 0.5 / g;

	float t = one_minus_g2 / (1.0f - g + 2.0f * g * xi);
	float invertcdf = one_over_2g * (one_plus_g2 - t * t);
	float cosTheta = abs(g) < 0.001 ? 2 * xi - 1 : invertcdf;
	float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));
	vec3 t0, t1;
	create_ortho_basis(w_in, t0, t1);
    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * w_in;
}

vec3 hg_phase_sample(vec3 w_in, float g, out float pdf) {
	float phi = random() * 2 * pi;
    float xi = random();
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
    float one_over_2g = 0.5 / g;

	float t = one_minus_g2 / (1.0f - g + 2.0f * g * xi);
	float invertcdf = one_over_2g * (one_plus_g2 - t * t);
	float cosTheta = abs(g) < 0.001 ? 2 * xi - 1 : invertcdf;
	float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));
	pdf = 0.25 / pi * (one_minus_g2) / pow(one_plus_g2 - 2 * g * cosTheta, 1.5);
	vec3 t0, t1;
	create_ortho_basis(w_in, t0, t1);
    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * w_in;
}

#endif

float hg_phase_eval(float cos_theta, float g)
{
	if (abs(g) < 0.001)
		return 0.25 / pi;
    float g2 = g * g;
    float one_minus_g2 = 1.0 - g2;
    float one_plus_g2 = 1.0 + g2;
	return 0.25 / pi * (one_minus_g2) / pow(one_plus_g2 - 2 * g * cos_theta, 1.5);
}

float hg_phase_eval(vec3 w_in, vec3 w_out, float g)
{
    return hg_phase_eval(dot(w_in, w_out), g);
}


/*
==================================
              Maps
==================================
*/

// #define CONCAT(a,b) a##b
#define EXPAND(x) x
#define NOT_SUPPORTED(msg) PRINT("[Error] Not supported (msg) in kernel {%d}.", int_ptr(_this.data).data[0]);
#define ARRAY_SIZE(d) (d > 0 ? d : 1)
#define CONCAT(a, b) a##b
#define CONCAT3(a,b,c) a##_##b##_##c
#define HELPER_MAP_BUFFER_NAME(name) CONCAT(buffer_, name)
#define MAP_BUFFER_NAME(name) EXPAND(HELPER_MAP_BUFFER_NAME(name))

#define HELPER_BUFFER(name, codename) CONCAT3(buffer,codename,name)
#define BUFFER(name) HELPER_BUFFER(name, RDV_CODENAME)
#define BUFFER_DECL(name, size) layout(buffer_reference, scalar, buffer_reference_align=4) buffer BUFFER(name) { float data[size]; };

// used in signatures for automatic helper submap forward and backward shortcuts.
#define SUBMAP_FORWARD_NAME SUBMAP_NAME
#define BUILD_SUBMAP_NAME(name) CONCAT(name,_bw)
#define SUBMAP_BACKWARD_NAME BUILD_SUBMAP_NAME(SUBMAP_NAME)


#define CAST(map_type, map_ptr) map_type(MAP_BUFFER_NAME(map_type) (map_ptr))



#endif