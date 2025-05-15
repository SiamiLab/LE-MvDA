function [A, S_jr_2D, D_jr_2D] = lemvda(X_pca, X_train, y_train, labels, view_labels, num_classes, num_views, samples_per_view, k_pca, kNN)
% LEMVDA Computes the affinity matrix and scatter matrices for LE-MvDA
% Inputs:
%   X_pca            - PCA-reduced feature matrix (k_pca x total_samples)
%   X_train          - Training feature matrix (k_pca x num_train_samples)
%   y_train          - Labels for training samples (num_train_samples x 1)
%   labels           - Labels for all samples (total_samples x 1)
%   view_labels      - View indicators (total_samples x 1)
%   num_classes      - Number of classes (c)
%   num_views        - Number of views (v)
%   samples_per_view - Samples per class per view
%   k_pca            - PCA dimension
%   kNN              - Number of nearest neighbors for local scaling
%
% Outputs:
%   A                - Affinity matrix (total_samples x total_samples)
%   S_jr_2D          - Within-class scatter matrix (flattened)
%   D_jr_2D          - Between-class scatter matrix (flattened)

% Initialization
total_samples = size(X_pca, 2);
c = num_classes;
v = num_views;
tr_nij = samples_per_view;
tr_ni = zeros(c, 1);

% Reshape training data
exp_X_train = reshape(X_train, [k_pca, c, v, nij_train]);

% Construct affinity matrix
A = zeros(total_samples, total_samples);
for class_num = 0:num_classes-1
    for view_num1 = 1:num_views
        for view_num2 = 1:num_views
            indices_view1 = find((labels == class_num) & (view_labels == view_num1));
            indices_view2 = find((labels == class_num) & (view_labels == view_num2));
            X_view1 = X_pca(:, indices_view1);
            X_view2 = X_pca(:, indices_view2);

            distance2 = pdist2(X_view1', X_view2').^2;
            [sorted_dist, ~] = sort(distance2, 2);

            kNN_dist2_view1 = sorted_dist(:, min(kNN, size(sorted_dist, 2)));
            sigma_view1 = sqrt(kNN_dist2_view1);
            kNN_dist2_view2 = sorted_dist(:, min(kNN, size(sorted_dist, 1)));
            sigma_view2 = sqrt(kNN_dist2_view2);

            localscale = sigma_view1 * sigma_view2';
            affinity_matrix = exp(-distance2 ./ localscale);

            A(indices_view1, indices_view2) = affinity_matrix;
            if view_num1 ~= view_num2
                A(indices_view2, indices_view1) = affinity_matrix';
            end
        end
    end
end

% Count training samples per class
for i = 1:c
    for j = 1:v
        tr_ni(i) = tr_ni(i) + tr_nij(i, j);
    end
end
tr_n = sum(tr_ni);

% Compute mean vectors per class/view
M_ij = zeros(k_pca, c, v);
for i = 1:c
    for j = 1:v
        for k = 1:tr_nij(i)
            M_ij(:, i, j) = M_ij(:, i, j) + exp_X_train(:, i, j, k);
        end
        M_ij(:, i, j) = M_ij(:, i, j) / tr_nij(i, j);
    end
end

% Compute within-class scatter matrix S_jr
S_jr = zeros(k_pca, k_pca, v, v);
y = y_train;
P_w = cell(c, 1);
tr_ni = sum(tr_nij, 2);
for i = 1:c
    n_i = sum(tr_nij(i, :));
    P_w{i} = zeros(n_i, n_i);
    for k = 1:n_i
        for l = 1:n_i
            if y(k) == i && y(l) == i
                P_w{i}(k, l) = A(k, l) / n_i;
            else
                P_w{i}(k, l) = 0;
            end
        end
    end
end

for j = 1:v
    for r = 1:v
        Temp2 = zeros(k_pca, k_pca);
        for i = 1:c
            Temp = zeros(k_pca, k_pca);
            Temp1 = zeros(k_pca, k_pca);
            if j == r
                for k = 1:tr_nij(i, j)
                    for l = 1:tr_nij(i, j)
                        Pkl_w = P_w{i}(k, l);
                        xijk = exp_X_train(:, i, j, k);
                        Temp1 = Temp1 + Pkl_w * (xijk * xijk');
                    end
                end
            end
            for k = 1:tr_nij(i, j)
                for l = 1:tr_nij(i, r)
                    Pkl_w = P_w{i}(k, l);
                    xijl = exp_X_train(:, i, j, l);
                    xirk = exp_X_train(:, i, r, k);
                    Temp = Temp + Pkl_w * (xijl * xirk');
                end
            end
            Temp2 = Temp2 + (Temp1 - Temp);
        end
        S_jr(:, :, j, r) = Temp2;
    end
end

% Compute between-class scatter matrix D_jr
D_jr = zeros(k_pca, k_pca, v, v);
for j = 1:v
    for r = 1:v
        Temp = zeros(k_pca, k_pca);
        Temp1 = zeros(k_pca, 1);
        Temp2 = zeros(k_pca, 1);
        for i = 1:c
            Temp = Temp + ((tr_nij(i, j) * tr_nij(i, r)) / tr_ni(i)) * ...
                (M_ij(:, i, j) * M_ij(:, i, r)');
            Temp1 = Temp1 + tr_nij(i, j) * M_ij(:, i, j);
            Temp2 = Temp2 + tr_nij(i, r) * M_ij(:, i, r);
        end
        D_jr(:, :, j, r) = Temp - ((Temp1 * Temp2') / tr_n);
    end
end

% Flatten 4D S_jr and D_jr into 2D matrices
nv = k_pca * v;
S_jr_2D = zeros(nv, nv);
D_jr_2D = zeros(nv, nv);

temp_r = 1;
temp_c = 1;
for j = 1:v
    for r = 1:v
        S_jr_2D(temp_r:(temp_r + k_pca - 1), temp_c:(temp_c + k_pca - 1)) = S_jr(:, :, j, r);
        D_jr_2D(temp_r:(temp_r + k_pca - 1), temp_c:(temp_c + k_pca - 1)) = D_jr(:, :, j, r);
        temp_c = temp_c + k_pca;
    end
    temp_c = 1;
    temp_r = temp_r + k_pca;
end
