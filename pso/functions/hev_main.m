%%HEV_MAIN
%
% Author(s):	Olle Sundstroem
%
% Version:	1.1 (19-Jun-2013)
%
% Institute for Dynamic Systems and Control, Department of Mechanical and
% Process Engineering, ETH Zurich
%
% This Source Code Form is subject to the terms of the Mozilla Public License,
% v. 2.0. If a copy of the MPL was not distributed with this file, You can
% obtain one at http://mozilla.org/MPL/2.0/.

function f = hev_main(x)

% load driving cycle
load JN1015

% create grid
grd.Nx{1}    = 61; 
grd.Xn{1}.hi = 0.7; 
grd.Xn{1}.lo = 0.4;

grd.Nu{1}    = 21; 
grd.Un{1}.hi = 1; 
grd.Un{1}.lo = -1;	% Att: Lower bound may vary with engine size.

% set initial state
grd.X0{1} = 0.55;

% final state constraints
grd.XN{1}.hi = 0.56;
grd.XN{1}.lo = 0.55;

% define problem
prb.W{1} = speed_vector; % (661 elements)
prb.W{2} = acceleration_vector; % (661 elements)
prb.W{3} = gearnumber_vector; % (661 elements)
prb.Ts   = 1;
prb.N    = 660*1/prb.Ts + 1;

% set options
options = dpm();
options.UseLine = 1;
options.SaveMap = 1;
options.MyInf = 1000;
options.Iter = 5;
options.InputType = 'c';
options.FixedGrid = 0;
options.HideWaitbar = 0;

% store new gear ratios (if available) in model parameters
if nargin<1
	par.r_gear = [17 9.6 6.3 4.6 3.7 3.5];
else
	par.r_gear = x;
end

% optimize
[res, ~] = dpm(@hev,par,grd,prb,options);

% return fuel consumption (L/100 km)
f = sum(res.C{1})/750/(sum(speed_vector)/1000/100)*1000;

	
