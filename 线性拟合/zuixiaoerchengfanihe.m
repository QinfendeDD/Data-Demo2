clear
clc
x=[2,4,5,6,6.8,7.5,9,12,13.3,15];
y=[-10,-6.9,-4.2,-2,0,2.1,3,5.2,6.4,4.5];
[~,k]=size(x);
for n=1:9
    X0=zeros(n+1,k);
    for k0=1:k           %构造矩阵X0
        for n0=1:n+1
            X0(n0,k0)=x(k0)^(n+1-n0);
        end
    end
    X=X0';
    ANSS=(X'*X)\X'*y';
    for i=1:n+1          %answer矩阵存储每次求得的方程系数，按列存储
       answer(i,n)=ANSS(i);
   end
    x0=0:0.01:17;
    y0=ANSS(1)*x0.^n    ;%根据求得的系数初始化并构造多项式方程
    for num=2:1:n+1     
        y0=y0+ANSS(num)*x0.^(n+1-num);
    end
    subplot(3,3,n)
    plot(x,y,'*')
    hold on
    plot(x0,y0)
end
suptitle('不同次数方程曲线拟合结果，从1到9阶')
