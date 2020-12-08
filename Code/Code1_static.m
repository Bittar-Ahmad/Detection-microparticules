clearvars
close all
clc

%%%%%%%%%%%%%%%%%%%%%% Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frame = load('data_10_particles');
frame = frame.G; %data.G for 1 particle (cropped image) data.B for 10 particles

s = size(frame);


%%%%%%%%%


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = sqrt(s(1).^2 + s(2).^2); %diagonal of the frame
n = s(1)*s(2);
R=1;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%
disp('The data has been succesfully processed.')


%%%%%%%%%%%%%%%%%%%%%%% Variation totale %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = zeros(s(1), s(2), s(3));
for i = 1 : s(3)
    for j1 = 1:s(1)
        for j2 = 1:s(2)
            
            k1 = j1-1;
            k2 = j2-1;
            k3 = j1+1;
            k4 = j2+1;
            
            if(k1 == 0)
                k1 = 1;
            end
            
            if(k2 == 0)
                k2 = 1;
            end
            
            if(k3 == s(1)+1)
                k3 = s(1);
            end
            
            if(k4 == s(2)+1)
                k4 = s(2);
            end
           
            W(j1,j2, i) = sqrt((frame(j1,j2, i) - frame(k1,j2, i)).^2 ... 
            + (frame(j1,j2, i) - frame(j1,k2, i)).^2 ...
            + (frame(j1,j2, i) - frame(k3,j2, i)).^2 ...
            + (frame(j1,j2, i) - frame(j1,k4, i)).^2); 
        end
    end
end
disp('The total variation has been calculated.')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%


%%%%%%%%%%%%%%%%% Probleme de minimisation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('The minimisation problem is now under processing ... Please wait.')
cvx_begin
    obj = 0;
    variable alphaa
    variable betaa
    variable psii
    variable etaa(s(3))
    variable gammaa
    z = rand(s(3),1);
    
    for i = 1:s(3)
        for j = 1:n
            
            xj = mod(j-1,s(1))+1;
            yj = floor((j-1)/s(1))+1;
            
            obj = obj + (1/2)*(W(xj,yj, i)*(-2*xj*alphaa - 2*yj*betaa + psii - etaa(i) + xj^2 + yj^2));
        end
        i
    end
            minimize (obj)
            subject to
                -etaa <= 0
                etaa + (gammaa - z).^2 - R.^2 <= 0
                alphaa.^2 + betaa.^2 - psii <= 0
                psii - d.^2 <= 0
 
cvx_end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%


%%%%%%%%%%%%%%%%%%% Affichage en 2-D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_2D = mean(frame.^2, 3);
figure
imagesc(data_2D)
hold on 
plot(alphaa,betaa,'r*',...
    'LineWidth',5,...
    'MarkerSize',5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%