function [normalized_feature, max_feature, mean_feature, std_feature] = feature_normalized(original_feature, max_feature, mean_feature, std_feature)

[num_sample,num_feature] = size(original_feature);
normalized_feature = zeros(num_sample, num_feature);

for i=1:num_feature
%     % 方式1
%     normalized_feature(:,i) = original_feature(:,i)/max_feature(i);
%     % 方式2
%     normalized_feature(:,i) = (original_feature(:,i)-mean_feature(i))/max_feature(i);
    % 方式3
    normalized_feature(:,i) = (original_feature(:,i)-mean_feature(i))/std_feature(i);
end