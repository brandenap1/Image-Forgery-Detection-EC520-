%% EC520 Project : Forgery Detection
% Branden Applewhite and Kakit Wong

% Load image
addpath /Users/brandenapplewhite/Desktop/BMPImages
image = imread('Beach.bmp');
seashell = double(imread('seashell.png'));


S = im2double(image);
S_adj = imresize(S,0.25);
S_length = size(S_adj(:,1,1))
S_height = size(S_adj(1,:,1))

combined = imfuse(S_adj,seashell,'blend');
imshow(combined)

%%

[H,dev,res] = firpm(10, [0.0 0.45 0.55 1.0], [1.0 1.0 0.0 0.0]);

S_flt = convn(S_adj,H,'same');

% Creating Mask to Demosaic
R_flt = [0 0 0 0; 0 1 0 1; 0 0 0 0; 0 1 0 1];
R_fltfull = repmat(R_flt,[128 192]);

G_flt = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0];
G_fltfull = repmat(G_flt,[128 192]);

B_flt = [1 0 1 0; 0 0 0 0; 1 0 1 0; 0 0 0 0];
B_fltfull = repmat(B_flt,[128 192]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency domain low pass filtering (Ideal LPF)
% ABHILASH SINGH (2022). Frequency domain lowpass filtering on images (2-D domain)...
% (https://www.mathworks.com/matlabcentral/fileexchange/88597-frequency-domain-lowpass-filtering-on-images-2-d-domain), MATLAB Central File Exchange. Retrieved April 24, 2022.

figure(1)
subplot(3,3,1)
imshow(S_adj)
title('Original image f(x,y)')

% Ideal low pass filter %%%%%%%%%%%%%%%%%%%%%%%%%
Green_img= S_adj(:,:,2);
P = size(Green_img);
M = P(1);
N = P(2);
F=fft2(Green_img,M,N);

subplot(3,3,4)
imshow(abs(F));
title('F(u,v)')

u0 = 100; %remove freq greater than u0
u = 0:(M-1);
v = 0:(N-1);
idx = find(u>M/2);
u(idx) = u(idx) - M;
idy = find(v>N/2);
v(idy) = v(idy) - N;
[V,U] = meshgrid(v,u);
D = sqrt(U.^2+V.^2);
H = double(D<=u0);

% display
subplot(3,3,5)
imshow(abs(fftshift(H)),[]);
title('Ideal LPF H(u,v)')

G=H.*F;
g=(ifft2(G));
subplot(3,3,6)
imshow(g)
title('Ideal LPF Image [G(x,y)]')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mosaiced Channels
R_channel = S_adj(:,:,1) .* R_fltfull;
G_channel = g .* G_fltfull;
B_channel = S_adj(:,:,3) .* B_fltfull;

subplot(3,3,7)
imshow(R_channel)
title('Red Mosaiced channel')

subplot(3,3,8)
imshow(G_channel)
title('Green Mosaiced channel')

subplot(3,3,9)
imshow(B_channel)
title('Blue Mosaiced channel')

Image = zeros(size(S_adj));
Image(:,:,1) = R_channel;
Image(:,:,2) = G_channel;
Image(:,:,3) = B_channel;

figure(3)
imshow(Image)
title('Full Mosaiced Image')

% Create Filter Coefficient Matrices


Rcfa = repmat([1 0;0 0], [256 384]);
Gcfa = repmat([0 1;1 0], [256 384]);
Bcfa = repmat([0 0;0 1], [256 384]);
% Split data into ?hat ? variables
Rh = S_flt(:,:,1) .* Rcfa;
Gh = g .* Gcfa;
Bh = S_flt(:,:,3) .* Bcfa;



filter_R = [1 2 1; 2 4 2; 1 2 1]/4;

filter_G = [0 1 0; 1 4 1; 0 1 0]/4;

filter_B = [1 2 1; 2 4 2; 1 2 1]/4;


% Bilinear Interpolation
R_Bilinear = convn(Rh,filter_R, 'same');
G_Bilinear = convn(Gh,filter_G, 'same');
B_Bilinear = convn(Bh,filter_B, 'same');



figure(2)
subplot(3,3,1)
imshow(R_Bilinear)
title('Red Channel Interpolated (Bilinear)');

subplot(3,3,2)
imshow(G_Bilinear)
title('Green Channel Interpolated (Bilinear)');

subplot(3,3,3)
imshow(B_Bilinear)
title('Blue Channel Interpolated (Bilinear)');

SBi = zeros(size(S_adj));
SBi(:,:,1) = R_Bilinear;
SBi(:,:,2) = G_Bilinear;
SBi(:,:,3) = B_Bilinear;

% Smooth Hue Transition Interpolation
G_SmoothHue = convn(G_channel, filter_G , 'same');
R_SmoothHue = G_SmoothHue .* convn(R_channel./G_SmoothHue, filter_R, 'same');
B_SmoothHue = G_SmoothHue .* convn(B_channel./G_SmoothHue, filter_B, 'same');

subplot(3,3,4)
imshow(R_SmoothHue)
title('Red Channel Interpolated (Smooth Hue)');

subplot(3,3,5)
imshow(G_SmoothHue)
title('Green Channel Interpolated (Smooth Hue)');

subplot(3,3,6)
imshow(B_SmoothHue)
title('Blue Channel Interpolated (Smooth Hue)');

SHue = zeros(size(S_adj));
SHue(:,:,1) = R_SmoothHue;
SHue(:,:,2) = G_SmoothHue;
SHue(:,:,3) = B_SmoothHue;

eval_size = size(SBi) - 2;

% Generate Error Maps
error_Green = zeros(eval_size);
error_BilinearG = zeros(eval_size);
error_SmoothHueG = zeros(eval_size);
error_Fake = zeros(eval_size);

error_Green = error(S_adj,error_Green);
error_BilinearG = error(SBi,error_BilinearG);
error_SmoothHueG = error(SHue,error_SmoothHueG);
error_Fake = error(combined,error_Fake);

figure(4)
subplot(1,4,1)
imshow(error_BilinearG,[])
title('Error Map of Bilinear Green Channel')

subplot(1,4,2)
imshow(error_Green,[])
title('Error Map of Green Channel')

subplot(1,4,3)
imshow(error_SmoothHueG,[])
title('Error Map of Smooth Hue Green Channel')

subplot(1,4,4)
imshow(error_Fake,[])
title('Error Map of Fake Green Channel')

Prob_Green = Prob(error_Green,eval_size);
Prob_BilinearG = Prob(error_BilinearG,eval_size);
Prob_SmoothHueG = Prob(error_SmoothHueG,eval_size);
Prob_Fake = Prob(error_Fake,eval_size);

figure(5)
subplot(1,4,1)
imshow(Prob_Green);
title('Probability Map of Image Green Channel')
subplot(1,4,2)
imshow(Prob_BilinearG);
title('Probability Map of Bilinear Green Channel')
subplot(1,4,3)
imshow(Prob_SmoothHueG);
title('Probability Map of Smooth Hue Green Channel')
subplot(1,4,4)
imshow(Prob_Fake);
title('Probability Map of Fake Green Channel')

Probfft_Green = fftshift(fft2(Prob_Green));
Probfft_BilinearG = fftshift(fft2(Prob_BilinearG));
Probfft_SmoothHueG = fftshift(fft2(Prob_SmoothHueG));
Probfft_Fake = fft2(Prob_Fake);

figure(6)
subplot(1,4,1)
imshow(mat2gray(abs(Probfft_Green)))
title('Fourier Transform of Green Channel')
subplot(1,4,2)
imshow(mat2gray(abs(Probfft_BilinearG)))
title('Fourier Transform of Bilinear Green Channel')
subplot(1,4,3)
imshow(mat2gray(abs(Probfft_SmoothHueG)))
title('Fourier Transform of Smooth Hue Green Channel')
subplot(1,4,4)
imshow(mat2gray(abs(Probfft_Fake)))
title('Fourier Transform of Fake Green Channel')

function Error_Map = error(Image,Error_Map)
    for i = 2:511
        for j = 2:767
            Sum = 0.25*Image(i+1,j,2) + 0.25*Image(i-1,j,2) + 0.25*Image(i,j+1,2) + 0.25*Image(i,j-1,2);
            Error_Map(i-1,j-1) = abs(Image(i,j,2) - Sum);
        end
    end
end

function Probability_Map = Prob(error,eval_size)
    % Set Parameters
    stDevi = 30;
    p1 = 1/256;
    n_condition = 0.001;

    Probability_Map = zeros(eval_size);
    w = zeros(eval_size);

    error_squared = error.^2;


    while 1
        for i = 1:510
            for j = 1:766
                Prob_a = (1/(stDevi*sqrt(2*pi)));
                Prob_num = error_squared(i,j);
                Prob_den = 2*(stDevi^2);
                Probability_Map(i,j) = Prob_a * exp(-(Prob_num/Prob_den));
                w(i,j) = Probability_Map(i,j)/(Probability_Map(i,j)+p1);
            end
        end

        % Maximization Step
        stDev_num = 0;
        stDev_den = 0;

        for i = 1:510
            for j = 1:766
                stDev_num = stDev_num + (w(i,j)*error_squared(i,j));
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
end
    