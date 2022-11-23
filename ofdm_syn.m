clc;
clear;
%% 参数设置
% OFDM的调制参数设置
N_sc=64;      %一个OFDM符号中含有的子载波数
N_fft=64;            % FFT 长度
N_cp=16;             % 循环前缀长度、Cyclic prefix
N_symbo=N_fft+N_cp;        % 1个完整OFDM符号长度
N_frm= 32;  % 一个OFDM符号之中包含的子帧

% 调制方式批量生成设置
hub = ['abcdef']; % 调制序号
M_dict = struct('a','2','b','4','c','8','d','16','e','64','f','256') % 调制阶数的字典
mode_dict =  struct('a','bpsk','b','qpsk','c','8psk','d','16qam','e','64qam','f','256qam') %调制方式字典

% 信道条件设置
Sampling_rate = 100e3; %信道采样频率
Path_delay = [0,1.4e-6,3e-6,7e-6,10.4e-6,16.8e-6,17.3e-6]; %信道的多径延迟
path_gain   = [0,-1.5,-3.6,-7,-9.1,-12,-16.9]; %信道多径增益
max_doppler_shift = 10e2; %最大多普勒偏移

% 多径-多普勒信道的生成
rayleighchan = comm.RayleighChannel(...
    'SampleRate',Sampling_rate, ...
    'PathDelays',Path_delay, ...
    'AveragePathGains',path_gain, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',max_doppler_shift);

% 第一个for循环，主要为了遍历各种预设的调制方式 
for mm = hub
 
    mode = mode_dict.(mm); %选择调制模式
    M = str2num(M_dict.(mm));   %调制阶数
    SNR=-10;         %初始信噪比仿真信噪比          
    Nd=N_sc*log2(M);               % 数据总数
    num = 3000; % 一个调制方式的数据单位总量
  
    % 文件名设置
    filename =['./otfs_rice/a',num2str(mm-'a'+1),'_ofdm_',mode,'/']; %保存文件名设置
    FILE = [mode,'_'];
    snr = num2str(SNR);

    %调制方式的生成
    for nn = 1:num

        
         sig_rec = [];% 保存信号

         % DD 瑞利信道的生成
         rayleighchan = comm.RayleighChannel(...
        'SampleRate',Sampling_rate, ...
        'PathDelays',Path_delay, ...
        'AveragePathGains',path_gain, ...
        'NormalizePathGains',true, ...
        'MaximumDopplerShift',max_doppler_shift);
   
        % 子帧的生成
        for jj = 1: N_frm
            
           %% 基带数据数据产生
            P_data=randi([0 1],1,Nd);
           %% 调制
            % QPSK
            data_temp1= reshape(P_data,log2(M),[])';             %以每组2比特进行分组，M=4
            data_temp2= bi2de(data_temp1);                       %二进制转化为十进制
            % modu_data=pskmod(data_temp2,M,pi/M);                 % QPSK调制

            if M < 10 
                modu_data=pskmod(data_temp2,M,pi/M);         
            end
            % QAM调制的方式
            if M > 10
                modu_data=qammod(data_temp2,M,'UnitAveragePower',true); 
            end
            
            % 信号进行并串转换
            data = modu_data;     

            %% IFFT   
            % 信号做DFT
            ifft_data=ifft(data,N_fft)*sqrt(N_fft);
            %% 插入保护间隔、循环前缀
            Tx_cd=[ifft_data(N_fft-N_cp+1:end,:);ifft_data];%把ifft的末尾N_cp个数补充到最前面
            %% 并串转换
            Tx_data=reshape(Tx_cd,[],1);%由于传输需要
            %% 信道（通过多经瑞利信道）
             %生成信道
             
%             CUM4EST(sig_rec, 1, 2560, 0, 'biased', 0, 0
            
             % 信号经过信道，相当于作卷积的过程
            Tx_data= rayleighchan(Tx_data);
        %     Tx_data = conv(Tx_data,H);
            rx_channel=awgn(Tx_data,SNR,'measured');%添加高斯白噪声
            sig_rec = [sig_rec;rx_channel];
            end 
            savewords = [filename,FILE,num2str(nn),snr,'.mat'];
            if mod(nn,500)==0
                   nn    
            end
            if mod(nn,num/6) == 0
                SNR = SNR + 5
            end
            save(savewords,'sig_rec')
    end
end