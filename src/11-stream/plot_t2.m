clear; font_size=12;
close all;
load t2.txt;
figure;
subplot(1,2,1);
semilogx(t2(:,1),t2(:,2),'o-');
xlabel('Number of streams','fontsize',font_size);
ylabel('Total time','fontsize',font_size);
xlim([1,100]);
ylim([0,10]);
title('(a)');
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);

subplot(1,2,2);
semilogx(t2(:,1),t2(1,2)./t2(:,2),'o-');
xlabel('Number of streams','fontsize',font_size);
ylabel('Speedup factor','fontsize',font_size);
xlim([1,100]);
ylim([0,2.5]);
title('(b)');
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
set(gca,'xtick',[1,2,4,8,16,32,64]);

