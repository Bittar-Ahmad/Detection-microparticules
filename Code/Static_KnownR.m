clearvars
close all
clc

% La taille de la frame
N       = [512,512,22];

% Le volume dans lequel la particule est presente
ranges  = [76.800 76.800 7];

% le rayon de particule
r       = 1;

% nombre de particules dans la frame
n       = 10;

% Ajouter du bruit Gaussian
sigma_N = 0;

BMsp    = [1 1 1];

%La vitesse de la particule
sp      = zeros(n,3); %Static

%%%% Generer les coordonnees (x, y, z) de n particules %%%%%%%%%%%%%%%%%%%%
flag = 1;
while flag
    iCenter = [3+72.800*rand(n,2), 1+5*rand(n,1)];
    [a,b] = knnsearch(iCenter,iCenter,'k',n-1);
    if sum(sum((b(:,2:end)<2*r)))>0
        flag = 1;
    else
        flag = 0;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Generer la DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[C,B,cinfo] = particlesSIM(n, ranges, N, r,...
    'TM',1,...%static (=1)
    'SPBM',BMsp,...
    'SP',sp,... %Pour la vitesse de particule
    'dTM',0.5,... %Pour delta
    'CENTER',iCenter,... (coordonnees des particules)
    'sigma',sigma_N,... (pour le bruit Gaussian)
    'poisson',0); %Pour le bruit Poisson


DD2 = mean(B.^2, 3);
figure
imagesc(DD2)



% colormap(gray)
% 
% J = imcrop(DD2, [245, 262, 30,30]);
% figure
% imagesc(J)
% 


% data_1_particle = B(262:262+30, 244:244+30,  :);
% DD3 = mean(data_1_particle.^2, 3);
% figure
% imagesc(DD3)
% 
% data_10_particle = B(:,:,:);
% filename = 'data_10_particle.mat';
% save(filename)
% 
% filename = 'data_1_particle.mat';
% save(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
