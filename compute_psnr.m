function [psnr,sim]=compute_psnr(im1,im2)
%Read the grayscale images
ref = imread(im1);
rec = imread(im2);
%Compute the square-root of the MSE
imdff = double(ref) - double(rec);
imdff = imdff(:);
mse = sqrt(mean(imdff.^2));
%Compute the PSNR
psnr = 20*log10(255/mse);
%Compute the Structural Similarity 
sim = ssim(rec,ref);