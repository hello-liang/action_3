% EXAMPLE_1	 This script demonstrates PSO applied to the Ackley Problem [1]
%
%	find x = [x1,x2]' such that
%
%	min: f(x) = 20 + exp(1)
%			  - 20*exp(-0.2*sqrt((1/n).*sum(x.^2,2))) ...
%		      - exp((1/n).*sum(cos(2*pi*x),2))
%
%	subject to:   x1 <= x2
%				x1^2 <= 4*x2
%				  -2 <= x <= 2 
%
% [1] Ebbesen, Kiwitz and Guzzella "A Generic Particle Swarm Optimization
%	  Matlab Function", 2012 American Control Conference, June 27-29,
%     Montreal, CA.
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

addpath(genpath('functions'))

% EXAMPLE 1: ACKLEY PROBLEM
% Options
options = pso;
options.PopulationSize	= 24;
options.Vectorized		= 'on';
options.BoundaryMethod	= 'penalize';
options.PlotFcns		= @psoplotbestf;
options.Display			= 'iter';
options.HybridFcn		= @fmincon;

% Problem
problem	= struct;
problem.fitnessfcn	= @ackley;
problem.nvars		= 2;
problem.Aineq		= [ 1 -1];
problem.bineq		= 0;
problem.lb			= [-2 -2];
problem.ub			= [ 2  2];
problem.nonlcon		= @mynonlcon;
problem.options		= options;

% Optimize
[x,fval,exitflag,output] = pso(problem);
