import triton
import triton.language as tl


@triton.jit
def spread_charge_kernel(grid_ptr, grid_size_ptr: tl.const, charges_ptr: tl.const, data_ptr: tl.const, ti_ptr: tl.const,
                         num_particles: tl.constexpr, order: tl.constexpr, BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    point_index = tl.arange(0, BLOCKSIZE)
    mask = point_index < order*order*order
    gridx = tl.load(grid_size_ptr)
    gridy = tl.load(grid_size_ptr+1)
    gridz = tl.load(grid_size_ptr+2)
    for particle in tl.range(pid, num_particles, BLOCKSIZE):
        charge = tl.load(charges_ptr+particle)
        ix = point_index % order
        iy = (point_index//order) % order
        iz = point_index//(order*order)
        xindex = (tl.load(ti_ptr+3*particle)+ix) % gridx
        yindex = (tl.load(ti_ptr+3*particle+1)+iy) % gridy
        zindex = (tl.load(ti_ptr+3*particle+2)+iz) % gridz
        dx = tl.load(data_ptr+3*num_particles*ix+3*particle, mask=mask)
        dy = tl.load(data_ptr+3*num_particles*iy+3*particle+1, mask=mask)
        dz = tl.load(data_ptr+3*num_particles*iz+3*particle+2, mask=mask)
        add = charge*dx*dy*dz
        tl.atomic_add(grid_ptr + xindex*gridy*gridz + yindex*gridz + zindex, add, mask=mask)

@triton.jit
def interp_derivatives_kernel(pos_deriv_ptr, charge_deriv_ptr, grid_ptr: tl.const, grid_size_ptr: tl.const, data_ptr: tl.const, ddata_ptr: tl.const,
                              ti_ptr: tl.const, num_particles: tl.constexpr, order: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    particle = block_start + tl.arange(0, BLOCK_SIZE)
    mask = particle < num_particles
    gridx = tl.load(grid_size_ptr)
    gridy = tl.load(grid_size_ptr+1)
    gridz = tl.load(grid_size_ptr+2)
    tix = tl.load(ti_ptr+3*particle)
    tiy = tl.load(ti_ptr+3*particle+1)
    tiz = tl.load(ti_ptr+3*particle+2)
    pos_derivx = tl.zeros((BLOCK_SIZE,), tl.float32)
    pos_derivy = tl.zeros((BLOCK_SIZE,), tl.float32)
    pos_derivz = tl.zeros((BLOCK_SIZE,), tl.float32)
    charge_deriv = tl.zeros((BLOCK_SIZE,), tl.float32)
    for ix in tl.range(0, order):
        xindex = (tix+ix) % gridx
        for iy in tl.range(0, order):
            yindex = (tiy+iy) % gridy
            for iz in tl.range(0, order):
                zindex = (tiz+iz) % gridz
                g = tl.load(grid_ptr + xindex*gridy*gridz + yindex*gridz + zindex, mask=mask)
                dx = tl.load(data_ptr+3*num_particles*ix+3*particle, mask=mask)
                dy = tl.load(data_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                dz = tl.load(data_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                ddx = tl.load(ddata_ptr+3*num_particles*ix+3*particle, mask=mask)
                ddy = tl.load(ddata_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                ddz = tl.load(ddata_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                pos_derivx += ddx*dy*dz*g
                pos_derivy += dx*ddy*dz*g
                pos_derivz += dx*dy*ddz*g
                charge_deriv += dx*dy*dz*g
    tl.store(pos_deriv_ptr+3*particle, pos_derivx, mask=mask)
    tl.store(pos_deriv_ptr+3*particle+1, pos_derivy, mask=mask)
    tl.store(pos_deriv_ptr+3*particle+2, pos_derivz, mask=mask)
    tl.store(charge_deriv_ptr+particle, charge_deriv, mask=mask)
