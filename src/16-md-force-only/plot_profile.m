clear; close all; font_size=12;

t_total=10.6;
t_force=4.80;
t_memory=0.0996*4+0.0981*3;
t_others=t_total-t_force-t_memory;
y=[t_force,t_memory,t_others,t_total];
figure
bar(y);
ylabel('time per step (ms)');
set(gca,'xtick',[])
title('256000 atoms');
text(0.7,6,'force','fontsize',15);
text(1.7,2,'copy','fontsize',15);
text(2.7,6,'others','fontsize',15);
text(3.7,11.5,'total','fontsize',15);


