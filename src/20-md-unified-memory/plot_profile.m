clear; close all; font_size=12;

t_total=[5.18,4.98,4.31,2.74];
t_force=[4.80,4.60,3.93,2.37];
speed=25.6./t_total;

figure

subplot(1,2,1);
bar(t_force);
hold
ylabel('time for force per step (ms)');
set(gca,'xtick',[],'fontsize',12)
text(0.7,-0.2,'A');
text(1.7,-0.2,'B');
text(2.7,-0.2,'C');
text(3.7,-0.2,'D');
title('(a)');

subplot(1,2,2);
bar(speed);
hold
ylabel('speed (10^7 atom * step / second)');
set(gca,'xtick',[],'fontsize',12)
text(0.7,-0.4,'A');
text(1.7,-0.4,'B');
text(2.7,-0.4,'C');
text(3.7,-0.4,'D');
title('(b)');

