clear;
close all;

%% ���ݹ�һ�������Լ����ݵ���ʾ
load('ex7data2.mat'); 

% ʵ��������һ��
max_X = max(X); % ÿ�����������ֵ
mean_X = mean(X);   % ÿ�������ľ�ֵ
std_X = std(X); % ÿ�������ı�׼��
feature = feature_normalized(X, max_X, mean_X, std_X);

figure;
plot(feature(:,1),feature(:,2),'bo','MarkerSize', 3);
hold on;

%% ���ݵĳ�ʼ��
feature = X;
[m, n] = size(feature);
K = 3; % ������Ŀ
% ��ʼ����������
r = 1 + (m-1).*rand([K 1]); % �������һ����Χ��1~m��K*1�ľ���
r = floor(r);   % �Բ��������������
centroid = feature(r,:);    % ��ԭ���������л�ó�ʼ�ľ�������
max_iteration = 5; % ����������
c = zeros(m,1); % ��ʼ���������
data = zeros(m,n+1);
J = zeros(max_iteration,1); % �洢��ʧ����

%% K-means����ʵ�ֵĹ���
for i=1:max_iteration
    for a=1:m
        c(a) = index_distance(feature(a,:), centroid, K);   % ����ÿ��������������ĵľ��룬���ݾ��뽫��������Ϊĳһ���������ĵ��������
        data(a,:) = [feature(a,:) c(a)];
    end
    first_class = find(c==1);   % �ҵ��������Ϊ1�������е�����ֵ
    second_class = find(c==2);  % �ҵ��������Ϊ2�������е�����ֵ
    third_class = find(c==3);   % �ҵ��������Ϊ3�������е�����ֵ

    %�ò�ͬ����ɫ����ͬ�������ʾ
    figure;
    plot(centroid(:,1),centroid(:,2),'ro','MarkerSize', 10); hold on;
    plot(feature(first_class,1),feature(first_class,2),'ro','MarkerSize', 3); hold on;
    plot(feature(second_class,1),feature(second_class,2),'bo','MarkerSize', 3); hold on;
    plot(feature(third_class,1),feature(third_class,2),'go','MarkerSize', 3); hold off;

    % ������ʧ����
    sum = 0;
    for z=1:m
        sum = sum + norm(data(z,1:n)-centroid(data(z,3),:));
    end
    J(i) = sum;

    % ���¾�������
    for b=1:K
        centroid(b,:)=mean(feature(find(c==b),:));
    end

end

%% ������ʧ��������
figure;
x = 1:max_iteration;
plot(x,J,'-');