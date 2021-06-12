function y=msf(x,fk) 
%均方频率，x为幅值，fk为频率值
[m,]=size(x); 
q=0;a=0;
    for i=1:m 
        p=x(i); %幅值
        h=fk(i); %频率值
        q=q+sum((h^2)*p); 
        a=a+sum(p); 
    end
    y=q/a; 
end

