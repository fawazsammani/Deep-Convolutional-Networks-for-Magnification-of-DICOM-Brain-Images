scale = 2;
folder = 'E:\FAWAZ\CNN Magnification\My Network\training images';
filepath = dir(fullfile(folder,'*png'));
num = length(filepath);
for img = 1:num
rgb = strcat(folder, '\', 'Train (', num2str(img), ')','.png');
rgb_image = imread(rgb);
gray = rgb2gray(rgb_image);
im_label = modcrop(gray, scale);
[hei, wid] = size(im_label);
input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
imwrite(input, ['Input_' num2str(img) '.png']);
end
