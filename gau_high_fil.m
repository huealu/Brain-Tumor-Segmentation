%% -------------Gaussian High Pass Filter--------------------%% 
% Inputs:
%   +   fou_im: the fourier transform of the image
%   +   thresh: the cutoff circle radius
% Outputs:
%   +   ret_image: the filtered image
function ret_image = gau_high_fil(fou_im,thresh)

[row, column] = size(fou_im);
d0 = thresh;
d = zeros(row,column);
h = zeros(row,column);
boost_fac = 1.75;
for i = 1 : row
    for j = 1 : column
        d(i,j) =  sqrt((i - (row / 2)) ^ 2 + (j - (column / 2)) ^ 2);
    end
end


for i = 1 : row
    for j = 1 : column
        h(i,j) = 1 - exp(-((d(i,j) ^ 2) / ( 2 * (d0 ^ 2))));
        h(i,j) = (boost_fac - 1) + h(i,j);
    end
end

for i = 1 : row
    for j = 1 : column
        ret_image(i,j) = (h(i,j)) * fou_im(i,j);
    end
end



