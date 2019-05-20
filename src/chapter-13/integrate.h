#pragma once
#include "common.h"

void equilibration
(
    int Ne, int N, int MN, double *box,
    double T_0, double time_step, Atom *atom
);

void production
(
    int Np, int Ns, int N, int MN, double *box,
    double T_0, double time_step, Atom *atom
);

