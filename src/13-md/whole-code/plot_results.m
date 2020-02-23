clear;close all; font_size=12;
load energy.txt;

t=(1:size(energy,1))*0.5; %ps
figure;
plot(t,energy(:,1),'linewidth',2);
hold on;
plot(t,energy(:,2),'linewidth',2);
plot(t,sum(energy,2),'linewidth',2);
xlabel('time (ps)', 'fontsize', font_size);
ylabel('energy (eV)', 'fontsize', font_size);
set(gca,'fontsize', font_size);
legend('kinetic','potential','total');

figure;
plot(t,sum(energy,2)/sum(energy(1,:))-1,'linewidth',2);
xlabel('time (ps)', 'fontsize', font_size);
ylabel('E(t) / E(t=0) - 1', 'fontsize', font_size);
set(gca,'fontsize', font_size);


