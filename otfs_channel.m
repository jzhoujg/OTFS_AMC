clc;
clear;
%% 参数设置
N_ti = 16;
N_sc= 32;      %一个OFDM符号中含有的子载波数

M=16;               %调制阶数
SNR=10;         %仿真信噪比
N_frm=1;            % 每种信噪比下的仿真帧数、frame
Nd=N_ti*N_sc*log2(M);               % 数据总数
L = 1;                %信道长度
ERR = [];
K_rician = 20; 
mode = '16qam';
num = 1; %


Sampling_rate = 550e3;
Path_delay = [0,1.4e-6,3e-6,7e-6,10.4e-6,16.8e-6,17.3e-6];
path_gain = [0,-1.5,-3.6,-7,-9.1,-12,-16.9];
max_doppler_shift = 15e2;




rayleighchan = comm.RayleighChannel(...
    'SampleRate',Sampling_rate, ...
    'PathDelays',Path_delay, ...
    'AveragePathGains',path_gain, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',max_doppler_shift);


filename =['/otfs_',mode,'/'];
FILE = [mode,'_'];
snr = ['_',num2str(SNR),'dB'];

F_M = zeros(N_sc,N_sc);
cp_length = 8; % 循环前缀的长度
N_CP=zeros(cp_length,N_sc);

for i = 1:cp_length
    N_CP(i,i) = 1;
end
A_CP = [N_CP;eye(N_sc)];


for i_sc = 1: N_sc
    for i_ti = 1 : N_sc 
        F_M(i_sc,i_ti) =1/sqrt(N_sc)*exp (-1j * 2 * pi * i_sc *i_ti /N_sc );
    end
end

for nn = 1:num
    
    % 信道生成
    snr = ['_',num2str(SNR),'dB'];
     H = zeros(1,1);
     for l = 1 :L
        H(l) = 1/sqrt(2*L)*(randn(1)+ 1j*randn(1));
        H(l) = sqrt(K_rician/(K_rician+1)) + sqrt(1/(K_rician+1)) * H(l);
     end
     sig_rec = [];
    
    for jj = 1: N_frm
    
    %% 基带数据数据产生
        P_data=randi([0 1],1,Nd);

        %% 调制
        % QPSK
        data_temp1= reshape(P_data,log2(M),[])';             %以每组2比特进行分组，M=4
        data_temp2= bi2de(data_temp1);  
        %二进制转化为十进制
        
        if M < 10 
            modu_data=pskmod(data_temp2,M,pi/M); 
        
        end% QPSK调制
        if M > 10
            modu_data=qammod(data_temp2,M); 
        end
        

        modu_data=awgn(modu_data,SNR,'measured');
        
        modu_data = rayleighchan(modu_data);
        x_data =  real(modu_data);
        y_data =  imag(modu_data);
        
        scatter(x_data,y_data)
        

        % 16QAM
        % data_temp1= reshape(P_data,4,[])';             %以每组2比特进行分组，M=4
        % data_temp2= bi2de(data_temp1);                       %二进制转化为十进制
        % modu_data=qammod(data_temp2,16);                 % QPSK调制


        
    end
end



 %信号绘图
%  figure(1);
%  semilogy(SNR,Ber,'r-o');
%  legend('QPSK调制');
%  xlabel('SNR');
%  ylabel('BER');
%  title('Rayleigh信道下误比特率曲线');
% 
%  grid on;
