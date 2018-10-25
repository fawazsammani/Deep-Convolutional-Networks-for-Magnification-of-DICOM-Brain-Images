folder = 'C:\Users\user\Desktop\New folder';
filepath = dir(fullfile(folder,'*png'));
num = length(filepath);
for img = 1:num
rgb = strcat(folder, '\', 'Input (', num2str(img), ')','.png');
rgb_image = imread(rgb);
gray = rgb2gray(rgb_image);
imwrite(gray, ['Grayscale_' num2str(img) '.png']);
end