clear;close all;
load x1.txt;
load x2.txt;

figure;
histogram(x1,100);
xlabel('x','fontsize',12);
ylabel('number of counts','fontsize',12);
set(gca,'fontsize',12);

figure;
histogram(x2,100);
xlabel('x','fontsize',12);
ylabel('number of counts','fontsize',12);
set(gca,'fontsize',12);