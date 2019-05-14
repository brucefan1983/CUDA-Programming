#pragma once

static void apply_mic
(
    double lx, double ly, double lz, double lxh, double lyh, 
    double lzh, double *x12, double *y12, double *z12
)
{
    if (*x12 < - lxh)      { *x12 += lx; } 
    else if (*x12 > + lxh) { *x12 -= lx; }
    if (*y12 < - lyh)      { *y12 += ly; } 
    else if (*y12 > + lyh) { *y12 -= ly; }
    if (*z12 < - lzh)      { *z12 += lz; } 
    else if (*z12 > + lzh) { *z12 -= lz; }
}


