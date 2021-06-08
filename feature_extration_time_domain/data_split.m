clear all;
filename = 'S005R04 T0.csv'; % 输入文件名，注意文件和代码路径
x = readmatrix(filename);
Fs = 160;%采样率
nnum = size(x);%原始数据(对于S005R04 T0.csv  [10240,65] = [16*640，1+64])
num_sample = 640;%单个事件的采样点
num_channel = 64;%通道数
event = nnum(1) / num_sample; % 每一个事件占640个采样点

%时域部分*******************************************************************
x_n = zeros(event, num_sample, num_channel);
k_n = zeros(event, num_channel);
s_n = zeros(event, num_channel);
A_BP_n = zeros(event, num_channel);
for i = 1 : 1 : event
    x_n(i,:,:) = x(num_sample*(i-1)+1:num_sample*i, 2:(num_channel+1));%数据按事件划分，第一列为时间戳，舍去
    k_n(i,:) = kurtosis(x_n(i,:,:));%计算峭度
    s_n(i,:) = skewness(x_n(i,:,:));%计算偏度
    de_dimention = squeeze(x_n(i,:,:));%降维度,原本[1,640,64]到[640,64]。
    A_BP_n(i,:) = bandpower(de_dimention);%计算average bandpower
end

%对某一个事件进行采样，如可以加八个采样点,函数data_sample
sample = 8;%640/8
event1_sample = data_sample(x_n(1,:,:), sample);%x_n(1,:,:)为第一个事件
k_event1_n = zeros(sample, num_channel);
for j = 1 : 1 : sample
    k_event1_n(j,:) = kurtosis(event1_sample(j,:,:));
end
figure(3);
n = 0 : 1 : sample - 1;
plot(n, k_event1_n(:,1));title(['第一个事件', num2str(sample), '个采样点的峭度']);
%以上为样例说明

figure(1);
n = 0 : 1 : event - 1;
plot(n, k_n(:,1));title('峭度');
figure(2);
plot(n, s_n(:,1));title('偏度');
figure(4);
plot(n, A_BP_n(:,1:64));title('Average Bandpower');

%频域部分*******************************************************************
F_x_n = zeros(event, num_sample, num_channel);
for i=1 : 1 : event
    per_event_sample = squeeze(x_n(i,:,:));
    F_x_n(i,:,:) = fft(per_event_sample);%快速傅里叶变换
end
len = size(F_x_n);
F_x_n = F_x_n(:,1:floor(len(2) / 2),:);
n2 = 0 : 1 : num_sample/2 - 1;
n2 = 2*n2/len(2) * pi;
figure(5);
subplot(211);
plot(n2/pi, abs( F_x_n(1,:,6)));title('Fast Fourier Transformer')
xlabel('\omega/\pi');
subplot(212);
plot(n2/pi/2*Fs, abs( F_x_n(1,:,6)));title('Fast Fourier Transformer')
xlabel('Hz');

