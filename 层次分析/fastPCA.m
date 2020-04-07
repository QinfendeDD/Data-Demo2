%% 样本矩阵X，有8个样本，每个样本有4个特征，使用PCA降维提取k个主要特征（k<4）
k=2;                            %将样本降到k维参数设置
X=[1 2 1 1;                     %样本矩阵
      3 3 1 2; 
      3 5 4 3; 
      5 4 5 4;
      5 6 1 5; 
      6 5 2 6;
      8 7 1 2;
      9 8 3 7]
%% 使用Matlab工具箱princomp函数实现PCA
[COEFF SCORE latent]=princomp(X)
pcaData1=SCORE(:,1:k)            %取前k个主成分
% 参数说明：
    %1）COEFF 是主成分分量，即样本协方差矩阵的特征向量；
    %2）SCORE主成分，是样本X在低维空间的表示形式，即样本X在主成份分量COEFF上的投影 ，若需要降k维，则只需要取前k列主成分分量即可
    %3）latent：一个包含样本协方差矩阵特征值的向量；
    
     %% 自己实现PCA的方法
[Row Col]=size(X);
covX=cov(X);                                    %求样本的协方差矩阵（散步矩阵除以(n-1)即为协方差矩阵）
[V D]=eigs(covX);                               %求协方差矩阵的特征值D和特征向量V
meanX=mean(X);                                  %样本均值m
%所有样本X减去样本均值m，再乘以协方差矩阵（散步矩阵）的特征向量V，即为样本的主成份SCORE
tempX= repmat(meanX,Row,1);
SCORE2=(X-tempX)*V                              %主成份：SCORE
pcaData2=SCORE2(:,1:k)