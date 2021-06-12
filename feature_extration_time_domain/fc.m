function y=fc(x,fk)
%重心频率,x幅值，fk频率值
    [m,]=size(x);
    q=0;a=0;
    for i=1:m
        p=x(i);
        h=fk(i);
        q=q+sum(h*p);
        a=a+sum(p);
    end
    y=q/a;
end

