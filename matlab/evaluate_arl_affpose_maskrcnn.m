function  AVE_F_wb = evaluate_UMD(path)

% affordances index
aff_start=0+1;   % ignore {background} label
aff_end=1+11;   % change based on the dataset
aff_list = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

% get all files
list_gt = getAllFiles(path, '*_gt.png', 1);   % gWet all files in current folder
list_predicted = getAllFiles(path, '*_pred.png', 1);

list_predicted = sort(list_predicted);
list_gt = sort(list_gt); % make the same style

fprintf('gt: %d, pred: %d \n', length(list_gt), length(list_predicted));
assert(length(list_predicted)==length(list_gt)); % test length
num_of_files = length(list_gt);

F_wb_aff = nan(num_of_files,1);
F_wb_non_rank = [];
AVE_F_wb = 0;

% for aff_id = aff_start:aff_end  % from 2 --> final_aff_id
for aff_idx = 1:length(aff_list)  % from 2 --> final_aff_id
    aff_id = aff_list{aff_idx} + 1;
    for i=1:num_of_files
        
%         fprintf('------------------------------------------------\n');
%         fprintf('affordance id=%d, image i=%d \n', aff_id, i);
%         fprintf('current pred: %s\n', list_predicted{i});
%         fprintf('current grth: %s\n', list_gt{i});
        
        %%read image      
        pred_im = imread(list_predicted{i}); 
        gt_im = imread(list_gt{i});

%         fprintf('size pred_im: %d \n', size(pred_im));
%         fprintf('size gt_im  : %d \n', size(gt_im));
        
        pred_im = pred_im(:,:,1);
        gt_im = gt_im(:,:,1);
       
        targetID = aff_id - 1; %labels are zero-indexed so we minus 1
        
        % only get current affordance
        pred_aff = pred_im == targetID;
        gt_aff = gt_im == targetID;
        
        if sum(gt_aff(:)) > 0 % only compute if the affordance has ground truth
            F_wb_aff(i,1) = WFb(double(pred_aff), gt_aff);  % call WFb function
        else
            %fprintf('no ground truth at i=%d \n', i);
        end
        
    end
    fprintf('Averaged F_wb for affordance id=%d is: %f \n', aff_id-1, nanmean(F_wb_aff));
    F_wb_non_rank = [F_wb_non_rank; nanmean(F_wb_aff)];
end
fprintf('\nAVE over all Affordance IDs: %f \n\n', nanmean(F_wb_non_rank));
AVE_F_wb = nanmean(F_wb_non_rank);

end
