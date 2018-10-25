folder = 'C:\Users\Admin\Desktop\paper files\results\out\out_modified_th20\scale 3 - modify';
filepath = dir(fullfile(folder,'*png'));
num = length(filepath);

for i = 1:num
    path = strcat(folder,'\','Out_', num2str(i),'.png');
    image = imread(path);
    image = padarray(image,[1 1]);
    imwrite(image, ['Out_' num2str(i) '.png']);
end
    