clear; close all;
N=[32,64,128,256,512,1024];
T=[3.88426 2.33144, 2.05883 2.06199  2.11750 2.28921];

figure;
semilogx(N,T,'-o','linewidth',2,'markersize',7);
xlabel('Block Size','fontsize',15);
ylabel('Time (s)','fontsize',15);
xlim([20,2000]);
set(gca,'fontsize',15,'xtick',10.^(1:5));
