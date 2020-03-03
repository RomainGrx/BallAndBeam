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

Flag = Flag_1;

michel = -1;
% michel = 0;
% michel = 1;

if michel == 1
    u1 = -49;
elseif michel == 0
    u1 = 20;
else
    u1 = 49;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Copier dans LabVIEW jusqu'ici %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%