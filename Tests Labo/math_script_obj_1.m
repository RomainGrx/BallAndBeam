clear all
clear console
clc

% Normalement: Setpoint (reference) [cm]
% Pour cet objectif: angle de la poutre (alpha) que l'on veut [deg]
Ref = 0.15;

X = 0.03;    % Position [cm]
dt = 0.05;   % Periode d'echantillonnage [s]
u_1 = 9;     % Angle du servo a l'iteration precedente [deg]

% 8 flags que l'on peut passer et modifier d'iteration en iteration.
% Etat des flags a l'iteration precedente.
Flag_1 = [0 0 0 0 0 0 0 0];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Copier dans LabVIEW a partir d'ici %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rappel: liste des commandes valides en MathScript:
% https://zone.ni.com/reference/en-XX/help/373123C-01/lvtextmath/msfunc_classes/

% Objectif 1: mettre la poutre a un angle constant.
% Rappel: le servo a un angle positif quand il va en sens anti-horlogique
% Donc la poutre a un angle positif quand elle va en sens *horlogique*

% De base, Ref sert a exprimer la position en cm. Mais pour cet objectif,
% on va utiliser ref pour specifier l'angle de la poutre que l'on veut et
% ce en degres.

% Dimensions caracteristiques du systeme.
theta_offset = -1.07698393e-01;  % Offset (environ -6.17 deg)
d = 55 / 1000;   % Longueur de la premiere barre attachee au servo [m]
b = 150 / 1000;  % Distance entre le pivot et le point d'attache du second
                 % bras du servo [m]

% On utilise la formule qui lie l'angle theta (servo) a l'angle alpha
% (poutre) et puis voila.

theta_min = -50;
theta_max = 50;

% Si rad_to_deg et deg_to_rad ne marchent pas: utiliser rad2deg et deg2rad
theta = rad_to_deg(asin(d / b * sin(deg_to_rad(Ref)))) - theta_offset;

% Il devrait y avoir une securite sur la valeur de l'angle, mais on va
% en remettre une nous-memes juste pour etre sur de rien casser.
if theta < theta_min
    u = theta_min;
elseif theta > theta_max
    u = theta_max;
else
    u = theta;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Copier dans LabVIEW jusqu'ici %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%