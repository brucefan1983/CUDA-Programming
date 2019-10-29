clear;close all
load t.txt;
figure;
loglog(t(:,1),t(:,2),'o-','linewidth',2);
hold on;
loglog(t(:,1),t(:,3),'v-','linewidth',2);
xlabel('memory size (KB)','fontsize',12);
ylabel('time (ms)','fontsize',12);
set(gca,'fontsize',12);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('cudaMalloc()','cudaMemcpy()');