clear; close all; font_size=12;

nx=[4,8,16,32];
N=nx.^3*4;
t_force=[12.3355,12.5941,20.0845,159.348];
t_D2H=[0.28905,0.56733,2.31592,16.3199];
t_H2D=[0.19893,0.42692,1.74099,12.2404];
t_equilibration=[7.66749,8.27827,21.0492,164.19];
t_production=[7.82091,9.41774,26.8761,203.473];

figure
loglog(N,t_force,'d-');
hold on;
plot(N,t_D2H+t_H2D,'s-');
plot(N,t_force+t_D2H+t_H2D,'o-');
plot(N,t_equilibration+t_production,'v-');
legend('Force kernel','Memory transfer','GPU part','Whole program');


