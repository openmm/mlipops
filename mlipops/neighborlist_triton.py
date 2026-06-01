import triton
import triton.language as tl


@triton.jit
def find_sort_keys_kernel(key_ptr, positions_ptr: tl.const, grid_size_ptr: tl.const, bin_size, num_particles: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    particle = block_start + tl.arange(0, BLOCK_SIZE)
    mask = particle < num_particles
    x = (tl.load(positions_ptr+3*particle, mask=mask)/bin_size).floor().cast(tl.int32)
    y = (tl.load(positions_ptr+3*particle+1, mask=mask)/bin_size).floor().cast(tl.int32)
    z = (tl.load(positions_ptr+3*particle+2, mask=mask)/bin_size).floor().cast(tl.int32)
    x = tl.where(y%2 == 0, -x, x)
    y = tl.where(z%2 == 0, -y, y)
    gridx = tl.load(grid_size_ptr)
    gridy = tl.load(grid_size_ptr+1)
    key = x + y*gridx + z*gridx*gridy
    tl.store(key_ptr+particle, key, mask=mask)


@triton.jit
def find_neighbors_kernel(output_ptr, output_counter_ptr, block_pairs_ptr: tl.const, block_particles_ptr: tl.const, block_positions_ptr: tl.const,
                          box_vectors_ptr: tl.const, num_block_pairs, max_output_pairs, num_particles, cutoff2, include_self, include_symmetric):
    block_pair_index = tl.program_id(axis=0)
    block1 = tl.load(block_pairs_ptr+2*block_pair_index)
    block2 = tl.load(block_pairs_ptr+2*block_pair_index+1)
    x1 = tl.load(block_positions_ptr+32*3*block1+3*tl.arange(0, 32))
    y1 = tl.load(block_positions_ptr+32*3*block1+3*tl.arange(0, 32)+1)
    z1 = tl.load(block_positions_ptr+32*3*block1+3*tl.arange(0, 32)+2)
    x2 = tl.load(block_positions_ptr+32*3*block2+3*tl.arange(0, 32))
    y2 = tl.load(block_positions_ptr+32*3*block2+3*tl.arange(0, 32)+1)
    z2 = tl.load(block_positions_ptr+32*3*block2+3*tl.arange(0, 32)+2)
    pair_index = tl.arange(0, 1024)
    i = pair_index//32
    j = pair_index%32
    dx = x1.gather(i, 0)-x2.gather(j, 0)
    dy = y1.gather(i, 0)-y2.gather(j, 0)
    dz = z1.gather(i, 0)-z2.gather(j, 0)
    if box_vectors_ptr is not None:
        box_xx = tl.load(box_vectors_ptr)
        box_yx = tl.load(box_vectors_ptr+3)
        box_yy = tl.load(box_vectors_ptr+4)
        box_zx = tl.load(box_vectors_ptr+6)
        box_zy = tl.load(box_vectors_ptr+7)
        box_zz = tl.load(box_vectors_ptr+8)
        scale = tl.floor(dz/box_zz+0.5)
        dx -= scale*box_zx;
        dy -= scale*box_zy;
        dz -= scale*box_zz;
        scale = tl.floor(dy/box_yy+0.5)
        dx -= scale*box_yx;
        dy -= scale*box_yy;
        scale = tl.floor(dx/box_xx+0.5)
        dx -= scale*box_xx;
    particle1 = tl.load(block_particles_ptr+32*block1+i)
    particle2 = tl.load(block_particles_ptr+32*block2+j)
    include = (dx*dx + dy*dy + dz*dz < cutoff2) * (particle1 < num_particles) * (particle2 < num_particles)
    if not include_self:
        include *= particle1 != particle2
    if not include_symmetric:
        include *= 32*block1+i <= 32*block2+j
    num_to_save = include.sum()
    if num_to_save > 0:
        start_output = output_counter_ptr.atomic_add(num_to_save)
        if start_output+num_to_save < max_output_pairs:
            save_pos = include.cumsum()-1
            tl.store(output_ptr+2*(start_output+save_pos), particle1, mask=include)
            tl.store(output_ptr+2*(start_output+save_pos)+1, particle2, mask=include)
