import triton
import triton.language as tl


@triton.jit
def spread_charge_kernel(grid_ptr, grid_size_ptr: tl.const, charges_ptr: tl.const, data_ptr: tl.const, ti_ptr: tl.const,
                         batch_ptr: tl.const, num_particles: tl.constexpr, order: tl.constexpr, BLOCKSIZE: tl.constexpr):
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
        if batch_ptr is None:
            base_ptr = grid_ptr
        else:
            batch = tl.load(batch_ptr+particle)
            base_ptr = grid_ptr + batch*gridx*gridy*gridz
        tl.atomic_add(base_ptr + xindex*gridy*gridz + yindex*gridz + zindex, add, mask=mask)


@triton.jit
def spread_dipoles_kernel(grid_ptr, grid_size_ptr: tl.const, charges_ptr: tl.const, dipoles_ptr: tl.const,
                         data_ptr: tl.const, ddata_ptr: tl.const, ti_ptr: tl.const, batch_ptr: tl.const,
                         num_particles: tl.constexpr, order: tl.constexpr, BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    point_index = tl.arange(0, BLOCKSIZE)
    mask = point_index < order*order*order
    gridx = tl.load(grid_size_ptr)
    gridy = tl.load(grid_size_ptr+1)
    gridz = tl.load(grid_size_ptr+2)
    for particle in tl.range(pid, num_particles, BLOCKSIZE):
        charge = tl.load(charges_ptr+particle)
        dipolex = tl.load(dipoles_ptr+3*particle)
        dipoley = tl.load(dipoles_ptr+3*particle+1)
        dipolez = tl.load(dipoles_ptr+3*particle+2)
        ix = point_index % order
        iy = (point_index//order) % order
        iz = point_index//(order*order)
        xindex = (tl.load(ti_ptr+3*particle)+ix) % gridx
        yindex = (tl.load(ti_ptr+3*particle+1)+iy) % gridy
        zindex = (tl.load(ti_ptr+3*particle+2)+iz) % gridz
        dx = tl.load(data_ptr+3*num_particles*ix+3*particle, mask=mask)
        dy = tl.load(data_ptr+3*num_particles*iy+3*particle+1, mask=mask)
        dz = tl.load(data_ptr+3*num_particles*iz+3*particle+2, mask=mask)
        ddx = tl.load(ddata_ptr+3*num_particles*ix+3*particle, mask=mask)
        ddy = tl.load(ddata_ptr+3*num_particles*iy+3*particle+1, mask=mask)
        ddz = tl.load(ddata_ptr+3*num_particles*iz+3*particle+2, mask=mask)
        add = charge*dx*dy*dz + dipolex*ddx*dy*dz + dipoley*dx*ddy*dz + dipolez*dx*dy*ddz
        if batch_ptr is None:
            base_ptr = grid_ptr
        else:
            batch = tl.load(batch_ptr+particle)
            base_ptr = grid_ptr + batch*gridx*gridy*gridz
        tl.atomic_add(base_ptr + xindex*gridy*gridz + yindex*gridz + zindex, add, mask=mask)


@triton.jit
def interp_derivatives_kernel(pos_deriv_ptr, charge_deriv_ptr, grid_ptr: tl.const, grid_size_ptr: tl.const, data_ptr: tl.const, ddata_ptr: tl.const,
                              ti_ptr: tl.const, batch_ptr: tl.const, num_particles: tl.constexpr, order: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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
    if pos_deriv_ptr is not None:
        pos_derivx = tl.zeros((BLOCK_SIZE,), tl.float32)
        pos_derivy = tl.zeros((BLOCK_SIZE,), tl.float32)
        pos_derivz = tl.zeros((BLOCK_SIZE,), tl.float32)
    if charge_deriv_ptr is not None:
        charge_deriv = tl.zeros((BLOCK_SIZE,), tl.float32)
    if batch_ptr is None:
        base_ptr = grid_ptr
    else:
        batch = tl.load(batch_ptr+particle, mask=mask)
        base_ptr = grid_ptr + batch*gridx*gridy*gridz
    for ix in tl.range(0, order):
        xindex = (tix+ix) % gridx
        for iy in tl.range(0, order):
            yindex = (tiy+iy) % gridy
            for iz in tl.range(0, order):
                zindex = (tiz+iz) % gridz
                g = tl.load(base_ptr + xindex*gridy*gridz + yindex*gridz + zindex, mask=mask)
                dx = tl.load(data_ptr+3*num_particles*ix+3*particle, mask=mask)
                dy = tl.load(data_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                dz = tl.load(data_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                if pos_deriv_ptr is not None:
                    ddx = tl.load(ddata_ptr+3*num_particles*ix+3*particle, mask=mask)
                    ddy = tl.load(ddata_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                    ddz = tl.load(ddata_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                    pos_derivx += ddx*dy*dz*g
                    pos_derivy += dx*ddy*dz*g
                    pos_derivz += dx*dy*ddz*g
                if charge_deriv_ptr is not None:
                    charge_deriv += dx*dy*dz*g
    if pos_deriv_ptr is not None:
        tl.store(pos_deriv_ptr+3*particle, pos_derivx, mask=mask)
        tl.store(pos_deriv_ptr+3*particle+1, pos_derivy, mask=mask)
        tl.store(pos_deriv_ptr+3*particle+2, pos_derivz, mask=mask)
    if charge_deriv_ptr is not None:
        tl.store(charge_deriv_ptr+particle, charge_deriv, mask=mask)


@triton.jit
def interp_dipoles_kernel(phi_ptr, grid_ptr: tl.const, grid_size_ptr: tl.const, data_ptr: tl.const, ddata_ptr: tl.const,
                          d2data_ptr: tl.const, ti_ptr: tl.const, batch_ptr: tl.const, num_particles: tl.constexpr, order: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
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
    phi0 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi1 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi2 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi3 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi4 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi5 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi6 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi7 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi8 = tl.zeros((BLOCK_SIZE,), tl.float32)
    phi9 = tl.zeros((BLOCK_SIZE,), tl.float32)
    if batch_ptr is None:
        base_ptr = grid_ptr
    else:
        batch = tl.load(batch_ptr+particle, mask=mask)
        base_ptr = grid_ptr + batch*gridx*gridy*gridz
    for ix in tl.range(0, order):
        xindex = (tix+ix) % gridx
        for iy in tl.range(0, order):
            yindex = (tiy+iy) % gridy
            for iz in tl.range(0, order):
                zindex = (tiz+iz) % gridz
                g = tl.load(base_ptr + xindex*gridy*gridz + yindex*gridz + zindex, mask=mask)
                dx = tl.load(data_ptr+3*num_particles*ix+3*particle, mask=mask)
                dy = tl.load(data_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                dz = tl.load(data_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                ddx = tl.load(ddata_ptr+3*num_particles*ix+3*particle, mask=mask)
                ddy = tl.load(ddata_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                ddz = tl.load(ddata_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                d2dx = tl.load(d2data_ptr+3*num_particles*ix+3*particle, mask=mask)
                d2dy = tl.load(d2data_ptr+3*num_particles*iy+3*particle+1, mask=mask)
                d2dz = tl.load(d2data_ptr+3*num_particles*iz+3*particle+2, mask=mask)
                phi0 += dx*dy*dz*g
                phi1 += ddx*dy*dz*g
                phi2 += dx*ddy*dz*g
                phi3 += dx*dy*ddz*g
                phi4 += d2dx*dy*dz*g
                phi5 += dx*d2dy*dz*g
                phi6 += dx*dy*d2dz*g
                phi7 += ddx*ddy*dz*g
                phi8 += ddx*dy*ddz*g
                phi9 += dx*ddy*ddz*g
    tl.store(phi_ptr+10*particle, phi0, mask=mask)
    tl.store(phi_ptr+10*particle+1, phi1, mask=mask)
    tl.store(phi_ptr+10*particle+2, phi2, mask=mask)
    tl.store(phi_ptr+10*particle+3, phi3, mask=mask)
    tl.store(phi_ptr+10*particle+4, phi4, mask=mask)
    tl.store(phi_ptr+10*particle+5, phi5, mask=mask)
    tl.store(phi_ptr+10*particle+6, phi6, mask=mask)
    tl.store(phi_ptr+10*particle+7, phi7, mask=mask)
    tl.store(phi_ptr+10*particle+8, phi8, mask=mask)
    tl.store(phi_ptr+10*particle+9, phi9, mask=mask)
