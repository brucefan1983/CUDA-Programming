clear;close all;
% read in xy.txt
fid=fopen('xy.txt','r');
N=sscanf(fgetl(fid),'%d');
xy=zeros(N,2);
box=sscanf(fgetl(fid),'%f',[1,2]);
for n=1:N
    xy(n,:)=sscanf(fgetl(fid),'%f',[1,2]);
end
fclose(fid);
% read in neighbor1.txt
fid=fopen("neighbor1.txt",'r');
neighbor1=zeros(N,4);
for n=1:N
    tmp=sscanf(fgetl(fid),'%d').';
    tmp(2:end)=tmp(2:end)+1; % form C index to Matlab index
    neighbor1(n,1:tmp(1)+1)=tmp;
end
fclose(fid);
% plot
figure;
plot(xy(:,1),xy(:,2),'.');
hold on;
for n=1:N
    xy1=xy(n,:);
    for k=1:neighbor1(n,1)
        xy2=xy(neighbor1(n,1+k),:);
        %x=(xy1(1)+xy2(1))/2;
        %y=(xy1(2)+xy2(2))/2;
        %plot(x,y,'r.');
        x=linspace(xy1(1),xy2(1),10);
        y=linspace(xy1(2),xy2(2),10);
        plot(x,y,'r-');
    end
    if (mod(n,1000)==0);disp(n);end;
end
axis equal;
xlim([60,180]);
ylim([60,180]);
xlabel('x (angstrom)');
ylabel('y (angstrom)');
        