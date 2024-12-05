% Path to the main folder
baseFolder = '/home/genzellab/tresholding/CBD';

% Initialization of global results
resultsTable = [];
all_precision = zeros(1, 3);   % Precision for each state
all_recall = zeros(1, 3);      % Recall for each state
all_f1_score = zeros(1, 3);    % F1 score for each state
all_specificity = zeros(1, 3); % Specificity for each state
num_comparisons = 0;           % Counter for comparisons

% Recursive exploration of subfolders
allFiles = dir(fullfile(baseFolder, '**', '*.mat')); % Search all subfolders

% Identify automatic and manual files
autoFiles = allFiles(contains({allFiles.name}, 'resultats_ClusterStates_DetermineStates.mat'));
manualFiles = allFiles(contains({allFiles.name}, '-states_TS.mat') | endsWith({allFiles.name}, '-states_SM.mat') | ...
    endsWith({allFiles.name}, '-states_ES.mat') | endsWith({allFiles.name}, '-states_AS.mat') | ...
    endsWith({allFiles.name}, '-states.mat'));

% Compare corresponding files
for i = 1:length(autoFiles)
    autoFilePath = fullfile(autoFiles(i).folder, autoFiles(i).name);

    % Find manual files in the same folder
    folderFiles = manualFiles(strcmp({manualFiles.folder}, autoFiles(i).folder));

    for j = 1:length(folderFiles)
        manualFilePath = fullfile(folderFiles(j).folder, folderFiles(j).name);

        % Load the files
        try
            disp('Processing...');
            ManualState = load(manualFilePath);
            AutoState = load(autoFilePath);

            % Extract states
            data1 = ManualState.states; % Manual states
            data2 = AutoState.idx.states; % Automatic states (adapt according to your files)

            % Convert to column vectors
            data1 = data1(:);
            data2 = data2(:);

            % Check file lengths
            if length(data1) ~= length(data2)
                warning(['Files ' manualFilePath ' and ' autoFilePath ...
                         ' have different lengths. Skipped.']);
                continue;
            end

            % Filter valid states (1, 3, 5)
            valid_values = [1, 3, 5];

            % Compute metrics
            precision = zeros(1, 3);
            recall = zeros(1, 3);
            f1_score = zeros(1, 3);
            specificity = zeros(1, 3);

            for k = 1:length(valid_values)
                state = valid_values(k);

                TP = sum(data1 == state & data2 == state);
                TN = sum(data1 ~= state & data2 ~= state);
                PP = sum(data2 == state);
                P = sum(data1 == state);
                N = sum(data1 ~= state);

                % Calculate metrics
                precision(k) = TP / PP;
                recall(k) = TP / P;
                f1_score(k) = 2 * (precision(k) * recall(k)) / (precision(k) + recall(k));
                specificity(k) = TN / N;
            end

            % Replace NaN values with 0
            precision(isnan(precision)) = 0;
            recall(isnan(recall)) = 0;
            f1_score(isnan(f1_score)) = 0;
            specificity(isnan(specificity)) = 0;

            % Append results to the table
            resultsTable = [resultsTable; {manualFilePath, autoFilePath, ...
                precision, recall, f1_score, specificity}];

            % Accumulate results for global averages
            all_precision = all_precision + precision;
            all_recall = all_recall + recall;
            all_f1_score = all_f1_score + f1_score;
            all_specificity = all_specificity + specificity;
            num_comparisons = num_comparisons + 1;

        catch ME
            warning(['Error processing files: ' manualFilePath ' and ' autoFilePath]);
            disp(ME.message);
        end
    end
end

% Calculate global averages
if num_comparisons > 0
    avg_precision = all_precision / num_comparisons;
    avg_recall = all_recall / num_comparisons;
    avg_f1_score = all_f1_score / num_comparisons;
    avg_specificity = all_specificity / num_comparisons;

    % Display global results
    disp('Global Results (Average of all comparisons):');
    fprintf('Average Precision: Wake = %.2f, NREM = %.2f, REM = %.2f\n', avg_precision(1), avg_precision(2), avg_precision(3));
    fprintf('Average Recall: Wake = %.2f, NREM = %.2f, REM = %.2f\n', avg_recall(1), avg_recall(2), avg_recall(3));
    fprintf('Average F1 Score: Wake = %.2f, NREM = %.2f, REM = %.2f\n', avg_f1_score(1), avg_f1_score(2), avg_f1_score(3));
    fprintf('Average Specificity: Wake = %.2f, NREM = %.2f, REM = %.2f\n', avg_specificity(1), avg_specificity(2), avg_specificity(3));
else
    disp('No comparisons were made.');
end

% Display results as a table
figure('Color', 'w', 'Name', 'Performance Evaluation');
tiledlayout(1, 1);

% Create a matrix of metrics
metrics = [avg_precision; avg_recall; avg_f1_score; avg_specificity];

% Choose a colormap
colormap('default');

% Display the matrix of data with colors
imagesc(metrics, [0.4, 1]); % Limit colors between 0.4 and 1
colorbar;
axis equal;

% Add annotations on each cell
for row = 1:size(metrics, 1)
    for col = 1:size(metrics, 2)
        text(col, row, sprintf('%.2f', metrics(row, col)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    end
end

% Set X and Y axis labels
states = {'Wake', 'NREM', 'REM'}; % Example states for columns
metrics_labels = {'Precision', 'Recall', 'F1 Score', 'Specificity'}; % Labels for rows

% Add axis labels
set(gca, 'XTick', 1:3, 'XTickLabel', states, 'YTick', 1:4, ...
    'YTickLabel', metrics_labels, ...
    'FontSize', 12, 'FontWeight', 'bold');

% Add a title
title('Performance Evaluation', 'FontSize', 14, 'FontWeight', 'bold');
