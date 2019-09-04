clear; close all; fs=13;

N=[1,10,100,1000,10000,100000]*1000;
T=[0.0052160,0.0061760,0.022944,0.18519,1.7753,17.681];
speed_gpu=N./T;
speed_cpu=10^8/918.305;
ratio=speed_gpu/speed_cpu;

figure;

subplot(1,2,1);
loglog(N,T,'o-','linewidth',2,'markersize',7);
xlabel('N','fontsize',fs);
ylabel('Time (s)','fontsize',fs);
xlim([10^3,10^8]);
set(gca,'fontsize',fs,'xtick',10.^(3:8));
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(a)');

subplot(1,2,2);
loglog(N,ratio,'o-','linewidth',2,'markersize',7);
xlabel('N','fontsize',fs);
ylabel('Speedup factor','fontsize',fs);
xlim([10^3,10^8]);
set(gca,'fontsize',fs,'xtick',10.^(3:8));
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(b)');


B=[32,64,128,256,512,1024];
T=[33.719 19.567 17.684 17.757 18.638 20.253];

figure;
semilogx(B,T,'-o','linewidth',2,'markersize',7);
xlabel('Block Size','fontsize',fs);
ylabel('Time (s)','fontsize',fs);
xlim([20,2000]);
set(gca,'fontsize',fs,'xtick',10.^(1:5));
set(gca,'ticklength',get(gca,'ticklength')*2);




