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
