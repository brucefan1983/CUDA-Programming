#pragma once

static void __device__ apply_mic
(
    Box box, real *x12, real *y12, real *z12
)
{
    if      (*x12 < - box.lx2) { *x12 += box.lx; } 
    else if (*x12 > + box.lx2) { *x12 -= box.lx; }
    if      (*y12 < - box.ly2) { *y12 += box.ly; } 
    else if (*y12 > + box.ly2) { *y12 -= box.ly; }
    if      (*z12 < - box.lz2) { *z12 += box.lz; } 
    else if (*z12 > + box.lz2) { *z12 -= box.lz; }
}

