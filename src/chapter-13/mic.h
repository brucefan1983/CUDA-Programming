#pragma once

static void apply_mic
(
    double *box, double *x12, double *y12, double *z12
)
{
    if      (*x12 < - box[3]) { *x12 += box[0]; } 
    else if (*x12 > + box[3]) { *x12 -= box[0]; }
    if      (*y12 < - box[4]) { *y12 += box[1]; } 
    else if (*y12 > + box[4]) { *y12 -= box[1]; }
    if      (*z12 < - box[5]) { *z12 += box[2]; } 
    else if (*z12 > + box[5]) { *z12 -= box[2]; }
}


