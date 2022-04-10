%% EC520 Project : Forgery Detection
% Branden Applewhite and Kakit Wong

% Load image
addpath /Users/brandenapplewhite/Desktop
image = imread('bu2010.tif');


S = im2double(image);

R = S(:,:,1);
G = S(:,:,2);
B = S(:,:,3);

% Create CFA configurations for interpolation 

[H W] = size(image(:,:,1));

repmatH = H/2;
repmatW = W/2;


Rcfa = repmat([1 0;0 0], [repmatH repmatW]);
Gcfa = repmat([0 1;1 0], [repmatH repmatW]);
Bcfa = repmat([0 0;0 1], [repmatH repmatW]);

% Split data into 'hat' variables

Rh = S .* Rcfa;
Gh = S .* Gcfa;
Bh = S .* Bcfa;

% Create Filter Coefficient Matrices

filter_R = [1 2 1; 2 4 2; 1 2 1]/4;

filter_G = [0 1 0; 1 4 1; 0 1 0]/4;

filter_B = [1 2 1; 2 4 2; 1 2 1]/4;


% Bilinear Interpolation
R_Bilinear = convn(Rh,filter_R, 'same');
G_Bilinear = convn(Gh,filter_G, 'same');
B_Bilinear = convn(Bh,filter_B, 'same');



figure(1)
subplot(1,2,1)
imshow(S)
title('Original Image (Double)');

% subplot(1,4,2)
% imshow(R_Bilinear)
% title('Red Channel Interpolated (Bilinear)');

subplot(1,2,2)
imshow(G_Bilinear)
title('Green Channel Interpolated (Bilinear)');

% subplot(1,4,4)
% imshow(B_Bilinear)
% title('Blue Channel Interpolated (Bilinear)');

% Smooth Hue Transition Interpolation
G_SmoothHue = convn(Gh, filter_G , 'same');
R_SmoothHue = G_SmoothHue .* convn(Rh./G_SmoothHue, filter_R, 'same');
B_SmoothHue = G_SmoothHue .* convn(Bh./G_SmoothHue, filter_B, 'same');

figure(2)
subplot(1,2,1)
imshow(S)
title('Original Image (Double)');

% subplot(1,4,2)
% imshow(R_SmoothHue)
% title('Red Channel Interpolated (Smooth Hue)');

subplot(1,2,2)
imshow(G_SmoothHue)
title('Green Channel Interpolated (Smooth Hue)');

% subplot(1,4,4)
% imshow(B_SmoothHue)
% title('Blue Channel Interpolated (Smooth Hue)');

% Median Filter Interpolation
R_Median = convn(Rh, filter_R, 'same');
G_Median = convn(Gh, filter_G, 'same'); 
B_Median = convn(Bh, filter_B, 'same');

Mrg = R_Median - G_Median;
Mrb = R_Median - B_Median;
Mgb = G_Median - B_Median;

R_Median = S + Mrg.*Gcfa + Mrb.*Bcfa;
G_Median = S - Mrg.*Rcfa + Mgb.*Bcfa;
B_Median = S - Mrb.*Rcfa - Mgb.*Gcfa;

figure(3)
subplot(1,2,1)
imshow(S)
title('Original Image (Double)');

% subplot(1,4,2)
% imshow(R_Median)
% title('Red Channel Interpolated (Median Filter)');

subplot(1,2,2)
imshow(G_Median)
title('Green Channel Interpolated (Median Filter)');

% subplot(1,4,4)
% imshow(B_Median)
% title('Blue Channel Interpolated (Median Filter)');                   

% EM Algorithm

eval_size = size(image(:,:,1)) - 2;

error_Interpolated_Bilinear = zeros(eval_size);
error_Interpolated_SmoothHue = zeros(eval_size);
error_Interpolated_Median = zeros(eval_size);
error_Green = zeros(eval_size);

for i = 2:(eval_size(:,1)+1)
    for j = 2:(eval_size(1,:)+1)
        % Bilinear Interpolation Error
        Sum_Int_Bilinear = 0.25*G_Bilinear(i+1,j,2) + 0.25*G_Bilinear(i-1,j,2) + 0.25*G_Bilinear(i,j+1,2) + 0.25*G_Bilinear(i,j-1,2);
        error_Interpolated_Bilinear(i-1,j-1) = abs(G_Bilinear(i,j,2) - Sum_Int_Bilinear);
        % Smooth Hue Transition Interpolation Error
        Sum_Int_SmoothHue = 0.25*G_SmoothHue(i+1,j,2) + 0.25*G_SmoothHue(i-1,j,2) + 0.25*G_SmoothHue(i,j+1,2) + 0.25*G_SmoothHue(i,j-1,2);
        error_Interpolated_SmoothHue(i-1,j-1) = abs(G_Bilinear(i,j,2) - Sum_Int_SmoothHue);  
        % Median Filtered Interpolation Error
        Sum_Int_Median = 0.25*G_Median(i+1,j,2) + 0.25*G_Median(i-1,j,2) + 0.25*G_Median(i,j+1,2) + 0.25*G_Median(i,j-1,2);
        error_Interpolated_Median(i-1,j-1) = abs(G_Bilinear(i,j,2) - Sum_Int_Median);
        % Image Green Channel Error
        Sum_Green = 0.25*S(i+1,j,2) + 0.25*S(i-1,j,2) + 0.25*S(i,j+1,2) + 0.25*S(i,j-1,2);
        error_Green(i-1,j-1) = abs(S(i,j,2) - Sum_Green);
    end
end

figure(5)
subplot(2,2,1)
imshow(error_Green)
title('Error Map of Image Green Channel');

subplot(2,2,2)
imshow(error_Interpolated_Bilinear)
title('Error Map of Bilinear Interpolation (G)');

subplot(2,2,3)
imshow(error_Interpolated_SmoothHue)
title('Error Map of Smooth Hue Transition Interpolation (G)');

subplot(2,2,4)
imshow(error_Interpolated_Median)
title('Error Map of Median Filter Interpolation (G)');

% Expectation Step

% Set Parameters
stDevi = 30;
stDev = 0;
p1 = 1/256;
n_condition = 0.001;

Prob = zeros(eval_size);
w = zeros(eval_size);

error_Interpolated_Bilinear_Squared = error_Interpolated_Bilinear.^2;


while 1
    for i = 1:eval_size(1,:)
        for j = 1:eval_size(:,1)
            Prob_a = (1/(stDevi*sqrt(2*p1)));
            Prob_num = error_Interpolated_Bilinear_Squared(i,j);
            Prob_den = 2*(stDevi^2);
            Prob(i,j) = Prob_a * exp(-(Prob_num/Prob_den));
            w(i,j) = Prob(i,j)/(Prob(i,j)+p1);
        end
    end
    
    % Maximization Step
    
    for i = 1:eval_size(:,1)
        for j = 1:eval_size(1,:)
            stDev_num = stDev_num + (w(i,j)*error_Interpolated_Bilinear_Squared(i,j));
            stDev_den = stDev_den + w(i,j);
        end
    end
    
    stDev = sqrt(stDev_num/stDev_den);
    % Do while condition
    if ~(abs(stDev - stDevi)>n_condition)
        break;
    end
    
    stDevi = stDev;
end


figure(6)
imshow(mat2gray(Prob));

Prob_DFT = fft2(Prob);
figure(7)
imshow(mat2gray(real(Prob_DFT)));


