clear; close all;
N=[0.2, 0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000]*1000;
T=[0.0374, 0.0374, 0.0372,0.0381,0.0403,0.0514,0.0698,0.130,0.262,0.464,1.06,2.06,4.05,10.0];
speed_gpu=10000*N./T;
speed_cpu=10^6*1000/9.43;
ratio=speed_gpu/speed_cpu;

figure;
loglog(N,T,'o','linewidth',2,'markersize',7);
xlabel('N','fontsize',15);
ylabel('Time (s)','fontsize',15);
xlim([10^2,10^7]);
set(gca,'fontsize',15,'xtick',10.^(2:7));

figure;
loglog(N,ratio,'o','linewidth',2,'markersize',7);
xlabel('N','fontsize',15);
ylabel('Speedup factor','fontsize',15);
xlim([10^2,10^7]);
set(gca,'fontsize',15,'xtick',10.^(2:7));