clear;clf;hold on;
xy=load('xy.txt');
plot(xy(:,1),xy(:,2),'.');
axis equal;
box on
xlim([60,180]);
ylim([60,180]);
xlabel('x (angstrom)');
ylabel('y (angstrom)');