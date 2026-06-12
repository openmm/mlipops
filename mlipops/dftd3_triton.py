import triton
import triton.language as tl


@triton.jit
def compute_c6_kernel(c6_ptr, z_ptr: tl.const, cn_ptr: tl.const, cn_ref_ptr: tl.const, c6_ref_ptr: tl.const,
                      num_ref_ptr: tl.const, pairs_ptr: tl.const, num_pairs: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    pair_index = block_start + tl.arange(0, BLOCK_SIZE)
    mask = pair_index < num_pairs
    particle1 = tl.load(pairs_ptr+2*pair_index, mask=mask)
    particle2 = tl.load(pairs_ptr+2*pair_index+1, mask=mask)
    z1 = tl.load(z_ptr+particle1, mask=mask)
    z2 = tl.load(z_ptr+particle2, mask=mask)
    cn1 = tl.load(cn_ptr+particle1, mask=mask)
    cn2 = tl.load(cn_ptr+particle2, mask=mask)
    num_ref1 = tl.load(num_ref_ptr+z1, mask=mask)
    num_ref2 = tl.load(num_ref_ptr+z2, mask=mask)
    max_ref1 = tl.max(num_ref1)
    max_ref2 = tl.max(num_ref2)
    sum_c6 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_weight = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.range(max_ref1):
        cn_ref1 = tl.load(cn_ref_ptr+7*z1+i, mask=mask)
        for j in tl.range(max_ref2):
            cn_ref2 = tl.load(cn_ref_ptr+7*z2+j, mask=mask)
            ref_mask = (cn_ref1 >= 0)*(cn_ref2 >= 0)
            deltacn1 = cn1-cn_ref1
            deltacn2 = cn2-cn_ref2
            weight_ij = ref_mask*tl.exp(-4*(deltacn1*deltacn1 + deltacn2*deltacn2))
            sum_weight += weight_ij
            sum_c6 += weight_ij*tl.load(c6_ref_ptr+5096*z1+49*z2+7*i+j, mask=mask)
    tl.store(c6_ptr+pair_index, sum_c6/sum_weight, mask=mask)


@triton.jit
def backprop_c6_kernel(result_ptr, grad_output_ptr: tl.const, z_ptr: tl.const, cn_ptr: tl.const, cn_ref_ptr: tl.const, c6_ref_ptr: tl.const,
                      num_ref_ptr: tl.const, pairs_ptr: tl.const, num_pairs: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    pair_index = block_start + tl.arange(0, BLOCK_SIZE)
    mask = pair_index < num_pairs
    particle1 = tl.load(pairs_ptr+2*pair_index, mask=mask)
    particle2 = tl.load(pairs_ptr+2*pair_index+1, mask=mask)
    z1 = tl.load(z_ptr+particle1, mask=mask)
    z2 = tl.load(z_ptr+particle2, mask=mask)
    cn1 = tl.load(cn_ptr+particle1, mask=mask)
    cn2 = tl.load(cn_ptr+particle2, mask=mask)
    num_ref1 = tl.load(num_ref_ptr+z1, mask=mask)
    num_ref2 = tl.load(num_ref_ptr+z2, mask=mask)
    max_ref1 = tl.max(num_ref1)
    max_ref2 = tl.max(num_ref2)
    sum_c6 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_weight = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_dc61 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_dc62 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_dweight1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_dweight2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.range(max_ref1):
        cn_ref1 = tl.load(cn_ref_ptr+7*z1+i, mask=mask)
        for j in tl.range(max_ref2):
            cn_ref2 = tl.load(cn_ref_ptr+7*z2+j, mask=mask)
            ref_mask = (cn_ref1 >= 0)*(cn_ref2 >= 0)
            deltacn1 = cn1-cn_ref1
            deltacn2 = cn2-cn_ref2
            weight_ij = ref_mask*tl.exp(-4*(deltacn1*deltacn1 + deltacn2*deltacn2))
            sum_weight += weight_ij
            ref_value = tl.load(c6_ref_ptr+5096*z1+49*z2+7*i+j, mask=mask)
            sum_c6 += weight_ij*ref_value
            dweight1 = -8*weight_ij*deltacn1
            dweight2 = -8*weight_ij*deltacn2
            sum_dc61 += dweight1*ref_value
            sum_dc62 += dweight2*ref_value
            sum_dweight1 += dweight1
            sum_dweight2 += dweight2
    denom = 1/(sum_weight*sum_weight)
    grad = tl.load(grad_output_ptr+pair_index, mask=mask)
    grad_c6_1 = sum_dc61*grad
    grad_c6_2 = sum_dc62*grad
    grad_weight_1 = sum_dweight1*grad
    grad_weight_2 = sum_dweight2*grad
    total_grad_1 = (sum_weight*grad_c6_1 - sum_c6*grad_weight_1)*denom
    total_grad_2 = (sum_weight*grad_c6_2 - sum_c6*grad_weight_2)*denom
    tl.atomic_add(result_ptr+particle1, total_grad_1, mask=mask)
    tl.atomic_add(result_ptr+particle2, total_grad_2, mask=mask)
