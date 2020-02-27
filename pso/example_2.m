% EXAMPLE_2	 This script demonstrates PSO applied to the problem of
% optimizing gear ratios of a hybrid vehicle [1]
%
%	find x = [x1,x2,x3,x4,x5,x6]' such that
%
%	min: f(x|u*)
%
%	subject to:   x2 <= x1
%				  x3 <= x2
%				  x4 <= x3
%				  x5 <= x4
%				  x6 <= x5
%				xmin <= x <= xmax
%
% [1] Ebbesen, Kiwitz and Guzzella "A Generic Particle Swarm Optimization
% Matlab Function"
%
% Author(s):	Soren Ebbesen, 14-Sep-2011
%				sebbesen@idsc.mavt.ethz.ch
%
% Version:	1.1 (19-Jun-2013)
%
% Institute for Dynamic Systems and Control, Department of Mechanical and
% Process Engineering, ETH Zurich
%
% This Source Code Form is subject to the terms of the Mozilla Public License,
% v. 2.0. If a copy of the MPL was not distributed with this file, You can
% obtain one at http://mozilla.org/MPL/2.0/.


% WARNING_________________________________________________________________
% This is a non-vectorized version of example 2 in [1]. Thus, running this
% script as is may take a *long* time.

addpath(genpath('functions'))

% EXAMPLE 2: GEAR RATIO OPTIMIZATION
% Options
options = pso;
options.PopulationSize	= 24;
options.PlotFcns		= @psoplotbestf;
options.Display			= 'iter';
options.Vectorized		= 'off';
options.TolFun		    = 1e-6;
options.StallGenLimit   = 50;

% Problem
problem	= struct;
problem.fitnessfcn	= @hev_main;
problem.nvars		= 6;
problem.Aineq		= [-1  1  0  0  0  0
						0 -1  1  0  0  0
						0  0 -1  1  0  0
						0  0  0 -1  1  0
						0  0  0  0 -1  1];
problem.bineq		= zeros(size(problem.Aineq,1),1);
problem.lb			= [13.5,  7.6, 5.0, 3.9, 3.0, 2.8];
problem.ub			= [20.4, 11.5, 7.6, 5.5, 4.4, 4.2];
problem.options		= options;

% Optimize
[x,fval,exitflag,output] = pso(problem);
