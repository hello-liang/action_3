%%ACKLEY	Ackley's test function for numerical optimization
%
%	Known global minimum is the origin (in any dimension)
%
%	The function is vecotrized and supports any number of dimensions.
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


function f = ackley(x)

% Dimensions
n = size(x,2);

% Ackley's function (non-linear test function)
f = 20 + exp(1) ...
   -20*exp(-0.2*sqrt((1/n).*sum(x.^2,2))) ...
   -exp((1/n).*sum(cos(2*pi*x),2));