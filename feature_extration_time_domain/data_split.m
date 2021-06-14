filename = 'S005R04 T2.csv'; % �����ļ�����ע���ļ��ʹ���·��
x = readmatrix(filename);
Fs = 160;%������
nnum = size(x);%ԭʼ����(����S005R04 T0.csv  [10240,65] = [16*640��1+64])
num_sample = 640;%�����¼��Ĳ�����
num_channel = 64;%ͨ����
event = nnum(1) / num_sample; % ÿһ���¼�ռ640��������

%ʱ�򲿷�*******************************************************************
x_n = zeros(event, num_sample, num_channel);
k_n = zeros(event, num_channel);
s_n = zeros(event, num_channel);
A_BP_n = zeros(event, num_channel);
for i = 1 : 1 : event
    x_n(i,:,:) = x(num_sample*(i-1)+1:num_sample*i, 2:(num_channel+1));%���ݰ��¼����֣���һ��Ϊʱ�������ȥ
    k_n(i,:) = kurtosis(x_n(i,:,:));%�����Ͷ�
    s_n(i,:) = skewness(x_n(i,:,:));%����ƫ��
    de_dimention = squeeze(x_n(i,:,:));%��ά��,ԭ��[1,640,64]��[640,64]��
    A_BP_n(i,:) = bandpower(de_dimention);%����average bandpower
end

%��ĳһ���¼����в���������ԼӰ˸�������,����data_sample
sample = 8;%640/8
event1_sample = data_sample(x_n(1,:,:), sample);%x_n(1,:,:)Ϊ��һ���¼�
k_event1_n = zeros(sample, num_channel);
for j = 1 : 1 : sample
    k_event1_n(j,:) = kurtosis(event1_sample(j,:,:));
end
figure(3);
n = 0 : 1 : sample - 1;
plot(n, k_event1_n(:,1));title(['��һ���¼�', num2str(sample), '����������Ͷ�']);
%����Ϊ����˵��

figure(1);
n = 0 : 1 : event - 1;
plot(n, k_n(:,1));title('�Ͷ�');
figure(2);
plot(n, s_n(:,1));title('ƫ��');
figure(4);
plot(n, A_BP_n(:,1));title('Average Bandpower');

%Ƶ�򲿷�*******************************************************************
F_x_n = zeros(event, num_sample, num_channel);
for i=1 : 1 : event
    per_event_sample = squeeze(x_n(i,:,:));
    F_x_n(i,:,:) = fft(per_event_sample);%���ٸ���Ҷ�任
end
len = size(F_x_n);
F_x_n = F_x_n(:,1:floor(len(2) / 2),:);
n2 = 0 : 1 : num_sample/2 - 1;
n2 = 2*n2/len(2) * pi;
figure(5);
subplot(211);
plot(n2/pi, abs( F_x_n(5,:,8)));title('Fast Fourier Transformer')
xlabel('\omega/\pi');
subplot(212);
plot(n2/pi/2*Fs, abs( F_x_n(5,:,8)));title('Fast Fourier Transformer')
xlabel('Hz');

