clear;close all;
fid=fopen("xy.txt");
N=sscanf(fgetl(fid),'%d');
box=sscanf(fgetl(fid),'%f',[1,2]);
xy=zeros(N,2);
for n=1:N
    xy(n,:)=sscanf(fgetl(fid),'%f',[1,2]);
end
figure;
plot(xy(:,1),xy(:,2),'.');
axis equal;
xlim([60,180]);
ylim([60,180]);
xlabel('x (angstrom)');
ylabel('y (angstrom)');
