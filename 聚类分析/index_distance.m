function [c] = index_distance(feature, centroid, K)

distance = zeros(K,1);
for i=1:K
    distance(i) = norm(feature-centroid(i,:));
end

c = find(distance==min(distance));
end