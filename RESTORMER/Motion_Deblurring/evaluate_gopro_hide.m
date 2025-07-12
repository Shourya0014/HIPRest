// close all; clear all;

datasets = {'GoPro', 'HIDE'};
num_set = length(datasets);

tic
delete(gcp('nocreate'))
parpool('local', 20);

for idx_set = 1:num_set
    file_path = strcat('./results/', datasets{idx_set}, '/');
    gt_path = strcat('./Datasets/test/', datasets{idx_set}, '/target/');
    path_list = [dir(strcat(file_path, '*.jpg')); dir(strcat(file_path, '*.png'))];
    gt_list = [dir(strcat(gt_path, '*.jpg')); dir(strcat(gt_path, '*.png'))];
    
    % Limit to first 100 images (or fewer if dataset is small)
    img_num = min(100, length(path_list));  

    total_psnr = 0;
    total_ssim = 0;

    if img_num > 0 
        parfor j = 1:img_num
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path, image_name));
           gt = imread(strcat(gt_path, gt_name));

           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);

           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
        end
    end

    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;
    
    fprintf('For %s dataset (first %d images): PSNR: %f SSIM: %f\n', ...
            datasets{idx_set}, img_num, qm_psnr, qm_ssim);

end

delete(gcp('nocreate'))
toc
