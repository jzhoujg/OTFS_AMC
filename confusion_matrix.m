clear;
close all;
clc

% ��������
mat = [
 0.9990,   0.0010,   ;
 0.0003,   0.9997  ;
    ];


mat = mat
% ��ǩ
label = {'OFDM','OTFS'};

% ��������������ɫ
% ��ͨ������ʰɫ�����rgbɫֵ
maxcolor = [191,54,12]; % ���ֵ��ɫ
mincolor = [255,255,255]; % ��Сֵ��ɫ

% ����������
m = length(mat);
imagesc(1:m,1:m,mat)
xticks(1:m)
xlabel('Predict class','fontsize',13.5)
xticklabels(label)
yticks(1:m)
ylabel('Actual class','fontsize',13.5)
yticklabels(label)

% ���콥��ɫ
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];

colormap(mymap)
colorbar()

% ɫ���������
for i = 1:m
    for j = 1:m
        text(i,j,num2str(mat(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',10);
    end
end

% ͼ��������ȿ�
ax = gca;
ax.FontName = 'Times New Roman';
title('OFDM&OTFS(SNR = 5 dB)')
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
axis square

% ����
saveas(gca,'m.png')