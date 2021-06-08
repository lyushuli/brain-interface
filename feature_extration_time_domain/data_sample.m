function [y] = data_sample(x, N)
x = squeeze(x);
nnum = size(x);
span = nnum(1) / N;
y = zeros(N, span, 64);
for i = 1 : 1 : N
    y(i,:,:) = x(span*(i-1)+1:span*i,:);
end
end