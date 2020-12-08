function [C,B,Cinfo] = particlesSIM(n,ranges,N,r,varargin)
%
% MatLab function which generates n particles of radius r in the volume of
% dimension ranges(1) X ranges(2) X ranges(3).
%
% MANDATORY INPUT
% n          = number of particles
% ranges     = 3D array containg the dimension of the volume
% N          = 3D array containg the number of voxel in each dimension
% r          = radius of the particles
%
% OPTIONAL INPUT
%
% sigma     = standard deviation of the Gaussian noise affecting the data;
%             default: 0;
% poisson   = flag indicating if Poisson noise affects data; default: 0
% TM        = movement time; default: 1 (it means we acquire data for
%             a static particle)
% dTM       = delta of movement time; default: 0.01;
% dTA       = delta for acquisition time; default: 0.01;
% sp        = speed of the particles; default: 1;
% H         = PSF for the blurr; default: Gaussian.
%
% OUTPUT
%
% C         = Center of the particles
% A         = 3D sparse array containg the voxel of the centers
% B         = 3D array containing the images stacked vertically
%

% Unpack data
rangeX  = ranges(1);
rangeY  = ranges(2);
rangeZ  = ranges(3);
Nx      = N(1);
Ny      = N(2);
Nz      = N(3);
dx      = rangeX/Nx;
dy      = rangeY/Ny;
dz      = rangeZ/Nz;


%% Defaults
sigma       = 0;
poisson     = 0;
TM          = 1;
dTM         = 0.01;
sp          = rand(1,3);
sp          = sp/norm(sp);
spBM        = [1 1 1];
C_init      = r + repmat([rangeX-2.2*r, rangeY-2.2*r, rangeX-2.2*r],n,1).*rand(n,3);
filt        = fspecial('gaussian',5,1) ;

if (nargin-length(varargin)) ~= 4
    error('Wrong number of required parameters');
end

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
    fprintf('!\n');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'SIGMA'
                sigma = varargin{i+1};
            case 'PSF'
                filt = varargin{i+1};
            case 'POISSON'
                poisson = varargin{i+1};
            case 'TM'
                TM = varargin{i+1};
            case 'DTM'
                dTM = varargin{i+1};
            case 'SPBM'
                spBM = varargin{i+1};
            case 'SP'
                sp = varargin{i+1};
            case 'CENTER'
                C_init = varargin{i+1};
        end
    end
end



if Nx~=Ny
    error('Not handled yet... Sorry')
end

% Data creation
dCT    = round(TM/dTM);
B      = zeros(N(1),N(2),N(3));
C      = zeros(dCT*n,5);

for k = 1:n
    C(k,1) = 1;
    C(k,2) = k;
    C(k,3:end) = C_init(k,:);
end

for t = 2:dCT
    C(((t-1)*n+1):t*n,1) = t;
    for k = 1:n
        C((t-1)*n+k,2) = k;
        cand = C((t-2)*n+k,3:end) + dTM*sp(k,:) + dTM*randn(size(C((t-2)*n+k,3:end)))*diag(spBM);
        % Check the intersection with the other particles
        prev = C(((t-2)*n+1):((t-2)*n+k-1),3:end);
        [~,dist] = knnsearch(prev,cand);
        while dist<2*r
            fprintf('cacca\n')
            cand = C((t-2)*n+k,3:end) + dTM*sp(k,:) + dTM*randn(size(C((t-2)*n+k,3:end)))*diag(spBM);
            [~,dist] = knnsearch(prev,cand);
        end
        C((t-1)*n+k,3:end) = cand;
    end  
end

% Benchmark for control
Cinfo = cell(n,dCT);
for t =1:dCT
    for k=1:n
        Cinfo{k,t} = struct('Pos',[],'SpanFrames',[],'Radii',[]);
    end
end

% Coordinates for circle discretization
[X,Y] = meshgrid(1:N(1),1:N(2));
data  = [];
for k = 1:N(3)
    data = [data; dx*X(:), dy*Y(:), k*dz*ones(N(1)*N(2),1)];
end

for t = 1:dCT
    fprintf('TIME: %d\n',t)
    V = 10*ones(prod(N),1);
    
    for k = 1:n
        dist                    = sum((data-kron(C((t-1)*n+k,3:end),ones(size(data,1),1))).^2,2)<r^2;
        V(dist)                 = 220;
    end
    V = reshape(V,N);
    V = imfilter(V,filt);
    
    for z = 1:Nz
        for k = 1:n
            % Check the intersection of the k--th sphere with the z--th plane
            l = abs(C((t-1)*n+k,end)-z*dz);
            if (l<r && abs(l-r)>1e-8)
                % Save the useful information about this center
                Cinfo{k,t}.Pos          = C((t-1)*n+k,3:end);
                Cinfo{k,t}.SpanFrames   = [Cinfo{k,t}.SpanFrames; z];
                Cinfo{k,t}.Radii        = [Cinfo{k,t}.Radii; sqrt(r^2-l^2)];       
            end
        end
        % Gaussian Noise
        if sigma>0
            noise    = randn(Nx,Nx);
            noise    = noise/norm(noise,'fro')*(1+norm(V(:,:,z),'fro'));
            V(:,:,z) = V(:,:,z) + sigma*noise;
            V(:,:,z) = V(:,:,z) + abs(min(min((V(:,:,z)))));
        end
    end
    % Poisson noise
    if poisson
        V = 1e12*imnoise(1e-12*V,'Poisson');
    end
    % Cut of the out--of--bounds values
    V(V>255) = 255;
    
    B(:,:,:) = V;
    
end
clc




