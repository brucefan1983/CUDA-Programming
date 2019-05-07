clear;close all;
load xy.txt;
figure;
plot(xy(:,1),xy(:,2),'.');
axis equal;
xlim([60,180]);
ylim([60,180]);
xlabel('x (Angstrom)');
ylabel('y (Angstrom)');
