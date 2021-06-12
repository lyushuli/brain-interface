function  y=vf(x,fk) 
%频率方差，x为幅值，fk为频率值，fc为重心频率
    [m,]=size(x); 
    FC=fc(x,fk);
    q=0;a=0;
%     for i=1:m
%         s=x(i,:); 
%         h=fk(i,:); 
%         l=FC(i,:); 
%         a=((h-l).^2).*s; 
%         b=sum(a); 
%         c=sum(s); 
%         t=b/c; 
%         y(i,:)=t; 
%     end
    for i=1:m
        p=x(i);  %幅值
        h=fk(i);  %频率值
        q=q+((h-FC)^2)*p; 
        a=a+sum(p);
    end
    y=q/a;
end
