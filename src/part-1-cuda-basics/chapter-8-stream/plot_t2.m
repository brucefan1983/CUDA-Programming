clear; font_size=12;
close all;
load t2.txt;
figure;
subplot(1,2,1);
plot(t2(:,1),t2(:,2),'o');
xlabel('number of streams','fontsize',font_size);
ylabel('total time','fontsize',font_size);
title('(a)');
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);

subplot(1,2,2);
plot(t2(:,1),t2(1,2)./t2(:,2),'o');
xlabel('number of streams','fontsize',font_size);
ylabel('speedup factor','fontsize',font_size);
title('(b)');
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);