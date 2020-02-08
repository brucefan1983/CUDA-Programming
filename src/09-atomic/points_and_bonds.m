clear;
clf;hold on;
xy=load('xy.txt');
neighbor = load('neighbor.txt');
bands = zeros(sum(neighbor(:,1))*3,2);
i = 1;
for n = 1:size(xy,1)
    for k=1:neighbor(n,1)
        bands(i,:) = xy(n,:); i=i+1;
        bands(i,:) = xy(neighbor(n,1+k)+1,:); i=i+1;
        bands(i,:) = [NaN,NaN]; i=i+1;
    end
end
plot(bands(:,1),bands(:,2),'-k')
plot(xy(:,1),xy(:,2),'.');
axis equal;
box on
xlim([60,180]);
ylim([60,180]);
xlabel('x (angstrom)');
ylabel('y (angstrom)');