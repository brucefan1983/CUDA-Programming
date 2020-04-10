clear;close all; font_size=12;
%load cpp/energy.txt;
%load force-only/energy.txt;
load whole-code/energy.txt;

t=(1:size(energy,1))*0.5; %ps
ek=energy(:,1);
ep=energy(:,2);
et=ek+ep;
figure;
plot(t,ek,'linewidth',2);
hold on;
plot(t,ep,'linewidth',2);
plot(t,et,'linewidth',2);
xlabel('time (ps)', 'fontsize', font_size);
ylabel('energy (eV)', 'fontsize', font_size);
set(gca,'fontsize', font_size);
legend('kinetic','potential','total');

figure;
plot(t,(et-mean(et))/abs(mean(et)),'linewidth',2);
xlabel('time (ps)', 'fontsize', font_size);
ylabel('Relative energy fluctuations', 'fontsize', font_size);
set(gca,'fontsize', font_size);

