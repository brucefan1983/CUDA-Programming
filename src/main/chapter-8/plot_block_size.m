clear;close all;
data=[
1024 11.362 41.344
512 9.0763 73.920
256 7.6096 142.40
128 7.3170 276.61
64 13.022 540.04];
figure;
loglog(data(:,1),data(:,2),'s','linewidth',2);
hold on;
plot(data(:,1),data(:,3)/1000,'o','linewidth',2);
plot(data(:,1),data(:,2)+data(:,3)/1000,'x','linewidth',2);
xlim([60,1100]);
ylim([0.02,20]);
xlabel('block size','fontsize',15);
ylabel('time (ms)','fontsize',15);
legend('kernel-1','kernel-2','total');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);