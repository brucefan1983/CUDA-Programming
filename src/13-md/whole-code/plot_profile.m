clear; close all; font_size=12;

t_total=5.18;
t_force=4.80;
t_integrate=0.15;
t_temp=0.07;
y=[t_force,t_integrate,t_temp,t_total];
figure
bar(y);
ylabel('time per step (ms)');
set(gca,'xtick',[])
title('256000 atoms');
text(0.7,5.2,'force','fontsize',15);
text(1.7,1,'integrate','fontsize',15);
text(2.7,0.5,'scale','fontsize',15);
text(3.7,5.5,'total','fontsize',15);


