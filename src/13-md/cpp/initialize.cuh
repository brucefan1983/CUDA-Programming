#pragma once
#include "common.cuh"

void initialize_position(int nx, real ax, Atom *atom);
void initialize_velocity(int N, real T_0, Atom *atom);

