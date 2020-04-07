clear;
close all;

%% 数据归一化处理以及数据的显示
load('ex7data2.mat'); 

% 实现特征归一化
max_X = max(X); % 每个特征的最大值
mean_X = mean(X);   % 每个特征的均值
std_X = std(X); % 每个特征的标准差
feature = feature_normalized(X, max_X, mean_X, std_X);

figure;
plot(feature(:,1),feature(:,2),'bo','MarkerSize', 3);
hold on;

%% 数据的初始化
feature = X;
[m, n] = size(feature);
K = 3; % 聚类数目
% 初始化聚类中心
r = 1 + (m-1).*rand([K 1]); % 随机生成一个范围在1~m的K*1的矩阵
r = floor(r);   % 对产生的随机数求整
centroid = feature(r,:);    % 从原来的数据中获得初始的聚类中心
max_iteration = 5; % 最大迭代次数
c = zeros(m,1); % 初始化类别索引
data = zeros(m,n+1);
J = zeros(max_iteration,1); % 存储损失函数

%% K-means聚类实现的过程
for i=1:max_iteration
    for a=1:m
        c(a) = index_distance(feature(a,:), centroid, K);   % 计算每个样本与聚类中心的距离，根据距离将样本划分为某一个聚类中心的类别索引
        data(a,:) = [feature(a,:) c(a)];
    end
    first_class = find(c==1);   % 找到聚类类别为1在数据中的索引值
    second_class = find(c==2);  % 找到聚类类别为2在数据中的索引值
    third_class = find(c==3);   % 找到聚类类别为3在数据中的索引值

    %用不同的颜色将不同的类别显示
    figure;
    plot(centroid(:,1),centroid(:,2),'ro','MarkerSize', 10); hold on;
    plot(feature(first_class,1),feature(first_class,2),'ro','MarkerSize', 3); hold on;
    plot(feature(second_class,1),feature(second_class,2),'bo','MarkerSize', 3); hold on;
    plot(feature(third_class,1),feature(third_class,2),'go','MarkerSize', 3); hold off;

    % 计算损失函数
    sum = 0;
    for z=1:m
        sum = sum + norm(data(z,1:n)-centroid(data(z,3),:));
    end
    J(i) = sum;

    % 更新聚类中心
    for b=1:K
        centroid(b,:)=mean(feature(find(c==b),:));
    end

end

%% 绘制损失函数曲线
figure;
x = 1:max_iteration;
plot(x,J,'-');