import triton
import triton.language as tl


@triton.jit
def backprop_delta_kernel(result_ptr, grad_output_ptr: tl.const, pairs_ptr: tl.const,
                          num_pairs: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    pair_index = block_start + tl.arange(0, BLOCK_SIZE)
    mask = pair_index < num_pairs
    particle1 = tl.load(pairs_ptr+2*pair_index, mask=mask)
    particle2 = tl.load(pairs_ptr+2*pair_index+1, mask=mask)
    gradx = tl.load(grad_output_ptr+3*pair_index, mask=mask)
    grady = tl.load(grad_output_ptr+3*pair_index+1, mask=mask)
    gradz = tl.load(grad_output_ptr+3*pair_index+2, mask=mask)
    tl.atomic_add(result_ptr+3*particle1, -gradx, mask=mask)
    tl.atomic_add(result_ptr+3*particle1+1, -grady, mask=mask)
    tl.atomic_add(result_ptr+3*particle1+2, -gradz, mask=mask)
    tl.atomic_add(result_ptr+3*particle2, gradx, mask=mask)
    tl.atomic_add(result_ptr+3*particle2+1, grady, mask=mask)
    tl.atomic_add(result_ptr+3*particle2+2, gradz, mask=mask)


@triton.jit
def batch_periodic_displacements_kernel(displacements_ptr, batch_ptr: tl.const, box_vectors_ptr: tl.const,
                                        num_deltas, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    delta_index = block_start + tl.arange(0, BLOCK_SIZE)
    mask = delta_index < num_deltas
    dx = tl.load(displacements_ptr+3*delta_index, mask=mask)
    dy = tl.load(displacements_ptr+3*delta_index+1, mask=mask)
    dz = tl.load(displacements_ptr+3*delta_index+2, mask=mask)
    system = tl.load(batch_ptr+delta_index, mask=mask)
    box_xx = tl.load(box_vectors_ptr+9*system, mask=mask)
    box_yx = tl.load(box_vectors_ptr+9*system+3, mask=mask)
    box_yy = tl.load(box_vectors_ptr+9*system+4, mask=mask)
    box_zx = tl.load(box_vectors_ptr+9*system+6, mask=mask)
    box_zy = tl.load(box_vectors_ptr+9*system+7, mask=mask)
    box_zz = tl.load(box_vectors_ptr+9*system+8, mask=mask)
    scale = tl.floor(dz/box_zz+0.5)
    dx -= scale*box_zx
    dy -= scale*box_zy
    dz -= scale*box_zz
    scale = tl.floor(dy/box_yy+0.5)
    dx -= scale*box_yx
    dy -= scale*box_yy
    scale = tl.floor(dx/box_xx+0.5)
    dx -= scale*box_xx
    tl.store(displacements_ptr+3*delta_index, dx, mask=mask)
    tl.store(displacements_ptr+3*delta_index+1, dy, mask=mask)
    tl.store(displacements_ptr+3*delta_index+2, dz, mask=mask)
