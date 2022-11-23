clc;
clear;
%% 参数设置

% 基本的调制方式的参数设置
N_ti = 8;       % 多普勒索引数；时间间隔
N_sc= 64;       % 时延索引数；一个OFDM符号中含有的子载波数
cp_length = 16; % 循环前缀的长度
num = 12000 ;    % 每个调制方式生成数目


% 文件批处理的方式
hub = ['abcdef'];   % 索引
N_frm=4;            % 每种信噪比下的仿真帧数、frame
M_dict = struct('a','2','b','4','c','8','d','16','e','64','f','256') % 调制阶数字典
mode_dict = struct('a','bpsk','b','qpsk','c','8psk','d','16qam','e','64qam','f','256qam') %调制模式字典

for mm =hub
   
    SNR=-10;                            %仿真信噪比
    M=str2num(M_dict.(mm));             %调制阶数
    mode = mode_dict.(mm);              %调制模式
    Nd=N_ti*N_sc*log2(M);               % 数据总数
    % L = 1;                              %信道长度          
    % K_rician = 20; 
    % 文件名设置
   
    
    filename =['./otfs_rice/b',num2str(mm-'a'+1),'_otfs_',mode,'/'];
    FILE = [mode,'_']; % 文件命名方式
    snr = ['_',num2str(SNR),'dB']; %SNR

    %% 信道生成
    Sampling_rate = 100e3; %信道的采样频率
    Path_delay = [0,1.4e-6,3e-6,7e-6,10.4e-6,16.8e-6,17.3e-6]; %信道延迟
    path_gain   = [0,-1.5,-3.6,-7,-9.1,-12,-16.9]; %信道增益
    max_doppler_shift = 10e2; %最大多普勒频移
    rayleighchan = comm.RayleighChannel(...
        'SampleRate',Sampling_rate, ...
        'PathDelays',Path_delay, ...
        'AveragePathGains',path_gain, ...
        'NormalizePathGains',true, ...
        'MaximumDopplerShift',max_doppler_shift);

    % OFDM发射机部分的发射生成
    F_M = zeros(N_sc,N_sc);
    N_CP=zeros(cp_length,N_sc);
    
    % CP生成序列的制作
    for i = 1:cp_length
        N_CP(i,i) = 1;
    end
    A_CP = [N_CP;eye(N_sc)];
   
    % IFFT变换的算子
    for i_sc = 0: N_sc-1
        for i_ti = 0 : N_sc-1 
            F_M(i_sc+1,i_ti+1) =sqrt(1/N_sc)*exp (-1j * 2 * pi * i_sc *i_ti /N_sc );
        end
    end
    
    % 
    for nn = 1:num
        % 参数初始化
        snr = ['_',num2str(SNR),'dB'];
        sig_rec = [];

         %信道生成
         rayleighchan = comm.RayleighChannel(...
        'SampleRate',Sampling_rate, ...
        'PathDelays',Path_delay, ...
        'AveragePathGains',path_gain, ...
        'NormalizePathGains',true, ...
        'MaximumDopplerShift',max_doppler_shift);
     
        for jj = 1: N_frm

        %% 基带数据数据产生
            P_data=randi([0 1],1,Nd); % 生成数据的
            %% 调制
            data_temp1= reshape(P_data,log2(M),[])';             %以每组2比特进行分组，M=4
            data_temp2= bi2de(data_temp1);  
            %二进制转化为十进制
            if M < 10 
                modu_data=pskmod(data_temp2,M,pi/M); 
            end% QPSK调制
            if M > 10
                modu_data=qammod(data_temp2,M,'UnitAveragePower',true); 
            end
            % 16QAM
            % data_temp1= reshape(P_data,4,[])';             %以每组2比特进行分组，M=4
            % data_temp2= bi2de(data_temp1);                       %二进制转化为十进制
            % modu_data=qammod(data_temp2,16);                 % QPSK调制
            
            data = reshape(modu_data,[N_sc,N_ti]);

            %% ISFFT
            ifft_temp=fft(data,N_sc,1)/sqrt(N_sc);
            isfft_data=ifft(ifft_temp,N_ti,2)*sqrt(N_ti);
            refe =ifft(data,N_ti,2)*sqrt(N_ti);
            
            
            
            %% IFFT 
            ifft_data  = F_M'*isfft_data; %本质上对列做了一个IFFT
            %% 插入保护间隔、循环前缀
             Tx_cd = A_CP * ifft_data;
            % Tx_cd=[ifft_data(N_fft-N_cp+1:end,:);ifft_data];%把ifft的末尾N_cp个数补充到最前面
            %% 并串转换
            Tx_data=reshape(Tx_cd,[],1);%由于传输需要
            
            %% 信道（通过多经瑞利信道）
            %生成信道
            % 信号经过信道，相当于作卷积的过程
            Tx_data = rayleighchan(Tx_data);
            rx_channel=awgn(Tx_data,SNR,'measured');%添加高斯白噪声
            sig_rec = [sig_rec;rx_channel]; %保存数据
        end 
        savewords = [filename,FILE,num2str(nn),snr,'.mat'];
        if mod(nn,500) == 0
            nn
        end
        if mod(nn,num/6) == 0
            SNR = SNR + 5
        end
        
        
        %save(savewords,'sig_rec');
    end
end
