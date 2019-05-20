#pragma once
#include "common.cuh"

void initialize_position(int nx, double ax, Atom *atom);
void initialize_velocity(int N, double T_0, Atom *atom);

