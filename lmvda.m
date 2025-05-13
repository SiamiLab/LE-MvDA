function [M_ij, S_jr_2D, D_jr_2D, A] = lmvda(c, v, s_r, s_c, k_pca, tr_nij, X_train, labels, view_labels, kNN)
% c: Number of classes
% v: Number of views
% s_r, s_c: Rows and columns of images
% k_pca: Reduced dimensionality
% tr_nij: Samples per class-view pair
% X_train: PCA-transformed training data
% labels, view_labels: Class and view labels for each sample
% kNN: Number of nearest neighbors for affinity calculation

% Total samples
total_samples = sum(tr_nij(:));

% Step 1: Build the Affinity Matrix
A = zeros(total_samples, total_samples);  % Preallocate affinity matrix

for class_num = 1:c
    for view_num1 = 1:v
        for view_num2 = 1:v
            % Get sample indices for the given class and view combinations
            indices_view1 = find((labels == class_num) & (view_labels == view_num1));
            indices_view2 = find((labels == class_num) & (view_labels == view_num2));

            % Extract samples
            X_view1 = X_train(:, indices_view1);
            X_view2 = X_train(:, indices_view2);

            % Compute pairwise distances and scaling factor
            distance2 = pdist2(X_view1', X_view2').^2;  % Pairwise squared distances
            [sorted_dist, ~] = sort(distance2, 2);

            % Calculate local scaling factors based on k-nearest neighbors
            kNN_dist2_view1 = sorted_dist(:, max(1, min(kNN, size(sorted_dist, 2))));
            kNN_dist2_view2 = sorted_dist(:, max(1, min(kNN, size(sorted_dist, 1))));
            sigma_view1 = sqrt(kNN_dist2_view1);
            sigma_view2 = sqrt(kNN_dist2_view2);
            localscale = sigma_view1 * sigma_view2';

            % Define affinity using Gaussian kernel
            affinity_matrix = exp(-distance2 ./ localscale);

            % Update global affinity matrix
            A(indices_view1, indices_view2) = affinity_matrix;
            if view_num1 ~= view_num2
                A(indices_view2, indices_view1) = affinity_matrix';
            end
        end
    end
end

% Total number of samples per class
tr_ni = sum(tr_nij, 2);
tr_n = sum(tr_ni);

% Step 2: Mean Calculation
M_ij = zeros(k_pca, c, v);
for i = 1:c
    for j = 1:v
        M_ij(:, i, j) = mean(X_train(:, i, j, 1:tr_nij(i, j)), 4);
    end
end

% Step 3: Within-class Scatter Matrix Calculation
S_jr = zeros(k_pca, k_pca, v, v);
for j = 1:v
    for r = 1:v
        Temp2 = zeros(k_pca, k_pca);
        for i = 1:c
            Temp = (tr_nij(i, j) * tr_nij(i, r) / tr_ni(i)) * (M_ij(:, i, j) * M_ij(:, i, r)');
            Temp1 = zeros(k_pca, k_pca);
            if j == r
                for k = 1:tr_nij(i, j)
                    x_ijk = X_train(:, i, j, k);
                    Temp1 = Temp1 + A(k, k) * (x_ijk * x_ijk');
                end
            end
            Temp2 = Temp2 + (Temp1 - Temp);
        end
        S_jr(:, :, j, r) = Temp2;
    end
end

% Step 4: Between-class Scatter Matrix Calculation
D_jr = zeros(k_pca, k_pca, v, v);
for j = 1:v
    for r = 1:v
        Temp = zeros(k_pca, k_pca);
        Temp1 = zeros(k_pca, 1);
        Temp2 = zeros(k_pca, 1);
        for i = 1:c
            Temp = Temp + (tr_nij(i, j) * tr_nij(i, r) / tr_ni(i)) * (M_ij(:, i, j) * M_ij(:, i, r)');
            Temp1 = Temp1 + tr_nij(i, j) * M_ij(:, i, j);
            Temp2 = Temp2 + tr_nij(i, r) * M_ij(:, i, r);
        end
        D_jr(:, :, j, r) = Temp - ((Temp1 * Temp2') / tr_n);
    end
end

% Step 5: Convert 4D matrices S_jr and D_jr into 2-D matrices S_jr_2D and D_jr_2D
nv = k_pca * v;
S_jr_2D = zeros(nv, nv);
D_jr_2D = zeros(nv, nv);
temp_r = 1;
temp_c = 1;

for j = 1:v
    for r = 1:v
        S_jr_2D(temp_r:temp_r+k_pca-1, temp_c:temp_c+k_pca-1) = S_jr(:, :, j, r);
        D_jr_2D(temp_r:temp_r+k_pca-1, temp_c:temp_c+k_pca-1) = D_jr(:, :, j, r);
        temp_c = temp_c + k_pca;
    end
    temp_c = 1;
    temp_r = temp_r + k_pca;
end

disp('LMvDA matrices and affinity matrix computed successfully!');
