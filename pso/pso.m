function varargout = pso(...
	fitnessfcn,nvars,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options)
%PSO    Constrained optimization using particle swarm optimization algorithm.
%   PSO attempts to solve problems of the form:
%       min F(X)  subject to:  A*X  <= B, Aeq*X  = Beq (linear constraints)
%        X                     C(X) <= 0, Ceq(X) = 0 (nonlinear constraints)
%                              LB <= X <= ub
%
%   X = PSO(FITNESSFCN,NVARS) finds a local unconstrained minimum X to the
%   FITNESSFCN using PSO. NVARS is the dimension (number of design
%   variables) of the FITNESSFCN. FITNESSFCN accepts a vector X of size
%   1-by-NVARS, and returns a scalar evaluated at X.
%
%   X = PSO(FITNESSFCN,NVARS,A,b) finds a local minimum X to the function
%   FITNESSFCN, subject to the linear inequalities A*X <= B.
%
%   X = PSO(FITNESSFCN,NVARS,A,b,Aeq,beq) finds a local minimum X to the
%   function FITNESSFCN, subject to the linear equalities Aeq*X = beq as
%   well as A*X <= B. (Set A=[] and B=[] if no inequalities exist.)
%
%   X = PSO(FITNESSFCN,NVARS,A,b,Aeq,beq,lb,ub) defines a set of lower and
%   upper bounds on the design variables, X, so that a solution is found in
%   the range lb <= X <= ub. Use empty matrices for lb and ub if no bounds
%   exist. Set lb(i) = -Inf if X(i) is unbounded below;  set ub(i) = Inf if
%   X(i) is unbounded above.
%
%   X = PSO(FITNESSFCN,NVARS,A,b,Aeq,beq,lb,ub,NONLCON) subjects the
%   minimization to the constraints defined in NONLCON. The function
%   NONLCON accepts X and returns the vectors C and Ceq, representing the
%   nonlinear inequalities and equalities respectively. PSO minimizes
%   FITNESSFCN such that C(X)<=0 and Ceq(X)=0. (Set lb=[] and/or ub=[] if
%   no bounds exist.)
%
%   X = PSO(FITNESSFCN,NVARS,A,b,Aeq,beq,lb,ub,NONLCON,options) minimizes
%   with the default optimization parameters replaced by values in the
%   structure OPTIONS. OPTIONS can be created by calling the PSO function
%   with no input arguments and a single output argument, i.e., options =
%   pso.
%
%   X = PSO(PROBLEM) finds the minimum for PROBLEM. PROBLEM is a structure
%   that has the following fields:
%       fitnessfcn: <Fitness function>
%            nvars: <Number of design variables>
%            Aineq: <A matrix for inequality constraints>
%            bineq: <b vector for inequality constraints>
%              Aeq: <Aeq matrix for equality constraints>
%              beq: <beq vector for equality constraints>
%               lb: <Lower bound on X>
%               ub: <Upper bound on X>
%          nonlcon: <nonlinear constraint function>
%          options: <Options structure created with options = pso>
%
%   [X,FVAL] = PSO(FITNESSFCN, ...) returns FVAL, the value of the fitness
%   function FITNESSFCN at the solution X.
%
%   [X,FVAL,EXITFLAG] = PSO(FITNESSFCN, ...) returns EXITFLAG which
%   describes the exit condition of PSO. Possible values of EXITFLAG and the
%   corresponding exit conditions are
%
%     1 Average change in value of the fitness function over
%        options.StallGenLimit generations less than options.TolFun and
%        constraint violation less than options.TolCon.
%	  2 Fitness limit reached and constraint violation less than
%	     options.TolCon.
%     3 The value of the fitness function did not change in
%        options.StallGenLimit generations and constraint violation less
%        than options.TolCon.
%     4 Magnitude of step smaller than machine precision and constraint
%        violation less than options.TolCon. This exit condition applies
%        only to nonlinear constraints.
%     5 Fitness limit reached and constraint violation less than
%        options.TolCon. 
%     0 Maximum number of generations exceeded.
%    -1 Optimization terminated by the output or plot function.
%    -2 No feasible point found.
%    -4 Stall time limit exceeded.
%    -5 Time limit exceeded.
%
%   [X,FVAL,EXITFLAG,OUTPUT] = PSO(FITNESSFCN, ...) returns a
%   structure OUTPUT with the following information:
%          problemtype: <Type of constraints>
%          generations: <Total generations, excluding HybridFcn iterations>
%            funccount: <Total function evaluations>
%        maxconstraint: <Maximum constraint violation>, if any
%              message: <PSO termination message>
%
%
%   Example:
%     Unconstrained minimization of 'ackleys' fitness function of
%     numberOfVariables = 2
%      x = pso(@ackleys,2)
%
%   An example with inequality constraints and lower bounds
%    A = [1 1; -1 2; 2 1];  b = [2; 2; 3];  lb = zeros(2,1);
%    [x,fval,exitflag] = pso(@ackleys,2,A,b,[],[],lb);
%
%     FITNESSFCN can also be an anonymous function:
%        x = pso(@(x) 3*sin(x(1))+exp(x(2)),2)
%
%   See also FITNESSFUNCTION, @
%
%
%   Author(s):	Soren Ebbesen, 13-Feb-2012
%				sebbesen@idsc.mavt.ethz.ch
%
%   Version: 0.6.0 (19-Jun-2013)
%
%   Institute for Dynamic Systems and Control, Department of Mechanical and
%   Process Engineering, ETH Zurich
%
%   This Source Code Form is subject to the terms of the Mozilla Public
%   License, v. 2.0. If a copy of the MPL was not distributed with this
%   file, You can obtain one at http://mozilla.org/MPL/2.0/.

% Display options including default values
if eq(nargin, 0) && eq(nargout, 0)
	fprintf('            PopInitRange: [ matrix | {[]} ]\n');
	fprintf('          PopulationSize: [ positive integer | {16} ]\n');
	fprintf('             Generations: [ positive integer | {100} ]\n');
	fprintf('               TimeLimit: [ positive scalar (seconds) | {inf} ]\n');
	fprintf('            FitnessLimit: [ scalar | {-inf} ]\n');
	fprintf('           StallGenLimit: [ positive integer | {50} ]\n');
	fprintf('          StallTimeLimit: [ positive scalar (seconds) | {inf} ]\n');
	fprintf('                  TolFun: [ positive scalar | {1e-6} ]\n');
	fprintf('                  TolCon: [ positive scalar | {1e-6} ]\n');
	fprintf('               HybridFcn: [ @fmincon | @fminsearch | @fminunc | {[]} ]\n');
	fprintf('                 Display: [ ''off'' | ''iter'' | {''final''} ]\n');
	fprintf('              OutputFcns: [ function_handle | {[]} ]\n');
	fprintf('                PlotFcns: [ function_handle | @psoplotbestf | {[]} ]\n');
	fprintf('       InitialPopulation: [ matrix | {[]} ]\n');
	fprintf('       InitialVelocities: [ matrix | {[]} ]\n');
	fprintf('       InitialGeneration: [ positive integer | {1} ]\n');
	fprintf('             PopInitBest: [ matrix | {[]} ]\n');
	fprintf('     CognitiveAttraction: [ positive scalar  | {0.5} ]\n');
	fprintf('        SocialAttraction: [ positive scalar  | {1.0} ]\n');
	fprintf('           VelocityLimit: [ vector | {[]} ]\n');
	fprintf('          BoundaryMethod: [ ''nearest'' | ''absorb'' | {''penalize''} ]\n');
	fprintf('              Vectorized: [ ''on'' | {''off''} ]\n');
	return
end

% Generate default options
if nargin < 10
	options.PopInitRange		= [];		% range of initial random seed
	options.PopulationSize		= 24;		% number of particles in swarm
	options.Generations			= 100;		% maximum number of generations
	options.TimeLimit			= inf;		% terminate if time limit is reached
	options.FitnessLimit		=-inf;		% terminate when fitness drops below this value
	options.StallGenLimit		= 50;		% not change more than TolFun over StallGenLimit Generations
	options.StallTimeLimit		= inf;		% terminate if change less than TolFun over StallTimeLimit
	options.TolFun				= 1e-6;		% PSO terminates if global best value does
	options.TolCon				= 1e-6;		% Acceptable constraint violation
	options.HybridFcn			= [];		% invoke fmincon after pso
	options.Display				= 'final';	% {'none'}|iter|final
	options.OutputFcns			= [];		% call function(s) after each iteration
	options.PlotFcns			= [];		% leave empty for no plot
	options.InitialPopulation	= [];		% user specified initial positions
	options.InitialVelocities	= [];		% user specified initial velocities
	options.InitialGeneration	= 1;		% initial generation
	options.PopInitBest			= [];		% specify initial personal best positions
	options.CognitiveAttraction	= 0.5;		% acceleration constant, cognitive
	options.SocialAttraction	= 1.0;		% acceleration constant, social
	options.VelocityLimit		= [];		% maximum velocity of particles
	options.BoundaryMethod		= 'penalize'; % constraint handling method
	options.Vectorized			= 'off';	% if fun is vectorized (faster)
end

% Return options if pso is called without arguments
if eq(nargin, 0)
	varargout{1} = options;
	return
end

if eq(nargin, 1)
	% fitnessfcn is a problem structure
	if isstruct(fitnessfcn)
		if ~isfield(fitnessfcn,'options')
			options = pso;
		else
			options = fitnessfcn.options;
		end
		
		% De-struct problem structure
		[fitnessfcn,nvars,Aineq,bineq,Aeq,beq,lb,ub,nonlcon] =...
			psooptimstruct(fitnessfcn);
	else % Single input and non-structure.
		error('optim:pso:invalidInput',['The input should be a structure'...
			' with valid fields or provide at least two arguments to PSO.']);
	end
end

% Anonymous function check
% s = functions(fitnessfcn);
% if strcmpi(s.type,'anonymous') && strcmpi(options.Vectorized,'on')
% 	warning('optim:pso:anonymous',['OPTIONS ''Vectorized'' changed to ''off'' '...
% 	'because fitness function is an anonymous function.'])
% 	options.Vectorized = 'off';
% end

% Check constraints________________________________________________________
if ~exist('Aineq','var')
	Aineq = [];
end

if ~exist('bineq','var')
	bineq = [];
end

if ~exist('Aeq','var')
	Aeq = [];
end

if ~exist('beq','var')
	beq = [];
end

if ~exist('lb','var')
	lb = [];
else
	lb = lb(:); % convert to column vector
end

if ~exist('ub','var')
	ub = [];
else
	ub = ub(:); % convert to column vector
end

if ~exist('nonlcon','var')
	nonlcon = [];
end

if size(options.VelocityLimit,1)>1
	options.VelocityLimit = options.VelocityLimit';
end

if ~isempty(nonlcon)
	problemtype = 'nonlinerconstr';
elseif ~isempty(Aineq) || ~isempty(Aeq)
	problemtype = 'linearconstraints';
elseif ~isempty(lb) || isempty(ub)
	problemtype = 'boundconstraints';
end

% if ~isempty(lb)
% 	if eq(numel(lb),1)
% 		lb = repmat(lb,nvars,1);
% 	elseif ~eq(nvars,numel(lb))
% 		error('optim:pso:ConstraintsDimensions', ...
% 			'lb must be a scalar or a vector of length nvars')
% 	end
% end
% 
% if ~isempty(ub)
% 	if eq(numel(ub),1)
% 		ub = repmat(ub,nvars,1);
% 	elseif ~eq(nvars,numel(ub))
% 		error('optim:pso:ConstraintsDimensions', ...
% 			'ub must be a scalar or a vector of length nvars')
% 	end
% end

% Error checks_____________________________________________________________
% Check Display options
if ~any(strcmpi(options.Display,{'off','final','iter'}))
	error(['Invalid value for OPTIONS parameter Display: must be'...
		' ''off'',''iter'', or ''final''.'])
end

% Check Vectorized options
if ~any(strcmpi(options.Vectorized,{'on','off'}))
	error(['Invalid value for OPTIONS parameter Vectorized: must be'...
		' ''on'',''off''.'])
end

% Check BoundaryMethod options
if ~any(strcmpi(options.BoundaryMethod,{'nearest','absorb','penalize'}))
	error(['Invalid value for OPTIONS parameter BoundaryMethod: must be'...
		' ''nearest'',''absorb'', or ''penalize''.'])
end

% Check HybridFcn options
if ~isempty(options.HybridFcn)
	if iscell(options.HybridFcn)
		HybridFcn = options.HybridFcn{1};
		if length(options.HybridFcn)==2
			HybridOptions = options.HybridFcn{2};
		end
	else
		HybridFcn = options.HybridFcn; 
	end
	
	if isa(HybridFcn,'function_handle')
		if ~any(strcmpi(func2str(HybridFcn),{'fmincon','fminsearch','fminunc'}))
			error(['The field HybridFcn must be empty or contain one of these strings:'...
				' ''fminsearch'', ''fmincon'', or ''fminunc''.'])
		end
	else
		error('Invalid value for OPTIONS parameter HybridFcn.')
	end
end



% Stability________________________________________________________________
% Perez et al. "Particle swarm approach for structural design optimization"
% Journal of computers and structures, 2007.
c1 = options.CognitiveAttraction;
c2 = options.SocialAttraction;
if (c1+c2)>=4
	warning('optim:pso:Stability',...
		['Stability not guarranteed: sum of cognitive and '...
		'social attraction not smaller than four.'])
end

% Necessary condition: (c1+c2)/2-1 < phi < 1
iw1 = 1-eps;
iw2 = (c1+c2)/2-1 + eps;

% Initialize_______________________________________________________________
% Generation counter
state.generation = options.InitialGeneration-1;
% Position and velocity
[state.pos,state.vel] =...
	psogenseed(nvars,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
% Personal best positions
state.P = [options.PopInitBest;
		   state.pos(size(options.PopInitBest,1)+1:end,:)];
% Probe initial seed
state.pbestval = psodistjob(fitnessfcn,state.P,options);
% Global best value
[state.gbestval,gbestind] = min(state.pbestval,[],1);
% Global best position
state.G = state.P(gbestind,1:nvars);
% Trajectory of global best
state.gtr = [state.G state.gbestval; nan(options.Generations,nvars+1)];

% Initial call to plot and output functions
state = psooutput(options,state,'init');

% Display
if strcmpi(options.Display,'iter')
	fprintf('\n\t\t\tBest\t    Max\n')
	fprintf('Iteration   f-count\tf(x)\t    constraint\n')
end

% Set exitflag: empty if PSO is running
exitflag = [];

% Initialize timers
t1 = clock;
t2 = t1;

% START PSO LOOP___________________________________________________________
for k = options.InitialGeneration:options.Generations
	
	% Iteration counter
	state.generation = k;
	
	% History of global best
	state.gtr(k+1,:) = [state.G state.gbestval];
			
	% Iniertia: linearly decreasing function with lower bound (iw2)
	phi = max(iw2,(iw2-iw1)/options.Generations*(k-1) + iw1);
	
	% Randon weights in (0,1)
	gamma = rand(options.PopulationSize,2);
	
	
	% BEGIN PSO ALGORITHM__________________________________________________
	N = options.PopulationSize;
	% Velocity update
	state.vel = phi*state.vel...
		+ diag(c1*gamma(:,1),0)*(state.P-state.pos)...
		+ diag(c2*gamma(:,2),0)*(repmat(state.G,N,1)-state.pos);
	
	if ~isempty(options.VelocityLimit)
		% Limit velocities to VelocityLimit
		state.vel = (abs(state.vel)>repmat(options.VelocityLimit,N,1)).*sign(state.vel)...
			.*repmat(options.VelocityLimit,N,1)...
			+ (abs(state.vel)<=repmat(options.VelocityLimit,N,1)).*state.vel;
	end
	
	% Position update
	state.pos = state.pos + state.vel;
	% END PSO ALGORITHM____________________________________________________
	
	
	
	% HANDLE CONSTRAINTS___________________________________________________
	% Boundary method 'absorb'
	if strcmpi(options.BoundaryMethod,'nearest')
		state =...
			psonearest(nvars,state,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
	elseif strcmpi(options.BoundaryMethod,'absorb')
		state =...
			psoabsorb(state,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
	end
		
	% Evaluate objective function for new positions
	pval = psodistjob(fitnessfcn,state.pos,options);
	
	% Evaluate constraints
	[flag, g] = psopenalize(...
		[state.pos; state.G],Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
	
	% Always fall back on 'penalize' in case 'absorb' fails
	pval = pval + flag(1:end-1)*realmax;
	
	% GET NEW GLOBAL AND PERSONAL BEST POSITIONS AND VALUES________________
	% Update personal best positions and values
	state.P(pval<state.pbestval,:) = state.pos(pval<state.pbestval,1:nvars);
	state.pbestval(pval<state.pbestval) = pval(pval<state.pbestval);
	
	% Find best of current generation
	[gval,ind] = min(pval,[],1);
	
	% Update global best position and value
	if gval<state.gbestval
		state.G = state.pos(ind,1:nvars);
		state.gbestval = gval;
	end
	
	% Plot current generation
	state = psooutput(options,state,'iter');
	
	% Write to command window if desired by user
	if strcmpi(options.Display,'iter')		
		fprintf('%6i %10i %12.4f %12.4f\n',...
			k,k*options.PopulationSize,state.gbestval,max(max(g(end,:))))
	end
	
	% Note time of last change larger than TolFun
	if state.gtr(k+1:end)-state.gtr(k,end)>options.TolFun;
		t2 = clock;
	end
	
	% CHECK CONDITIONS FOR TERMINATION_____________________________________
	% Terminate on Generations
	if eq(k, options.Generations)
		msg = 'maximum number of generations exceeded.';
		exitflag = 0;
	% Terminate on StallGenLimit
	elseif k>=options.StallGenLimit && abs(state.gtr(k+1,end)...
			- state.gtr(k+1-options.StallGenLimit,end)) < options.TolFun
		msg = ['Average cumulative change in value of the fitness '...
			'function over options.StallGenLimit generations less '...
			'than options.TolFun and constraint violation less than options.TolCon.'];
		exitflag = 1;
	% Terminate on TimeLimit
	elseif etime(clock,t1)>options.TimeLimit
		msg = 'Time limit exceeded.';
		exitflag = -5;
	% Terminate of FitnessLimit
	elseif state.gbestval<options.FitnessLimit
		msg = 'Fitness limit reached and constraint violation less than options.TolCon.';
		exitflag = 2;
	% Terminate on StallTimeLimit
	elseif etime(clock,t2)>options.StallTimeLimit
		msg = 'Stall time limit exceeded.';
		exitflag = -4;
	elseif true(state.StopFlag)
		msg = 'Optimization terminated by the output or plot function.';
		exitflag = -1;
	end
	
	% Exit if exitflag is no longer empty
	if ~isempty(exitflag)
		flag =...
			psopenalize(state.G,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
		if true(flag)
			msg = 'No feasible point found.';
			exitflag = -2;
		end
	
		if strcmpi(options.Display,'iter') || strcmpi(options.Display,'final')
			fprintf('Optimization terminated: %s\n',msg)
		end
		break
	end
	
end % END OF PSO LOOP______________________________________________________



% SWITCH TO HYBRID FUNCTION________________________________________________
if ~isempty(options.HybridFcn)
	funcname = func2str(HybridFcn);
	
	fprintf(['Switching to the hybrid optimization algorithm ('...
		upper(funcname) ').\n\n'])
	
	switch funcname
		case 'fmincon'
			if ~exist('HybridOptions','var')
				HybridOptions = optimset('Simplex','off','LargeScale','off',...
					'Algorithm','active-set','Display','none');
			end
			[state.G,state.gbestval,exitflag] =...
				fmincon(fitnessfcn,state.G,...
				Aineq,bineq,Aeq,beq,lb,ub,nonlcon,HybridOptions);
		case 'fminsearch'
			if ~exist('HybridOptions','var')
				HybridOptions = optimset('Display','none');
			end
			[state.G,state.gbestval,exitflag] =...
				fminsearch(fitnessfcn,state.G,HybridOptions);
		case 'fminunc'
			if ~exist('HybridOptions','var')
				HybridOptions = optimset('Display','none');
			end
			[state.G,state.gbestval,exitflag] =...
				fminunc(fitnessfcn,state.G,HybridOptions);
	end
	
	fprintf([upper(funcname) ' has terminated.\n'])
end

% Final call to output and plot functions
psooutput(options,state,'done');

% Set final outputs
varargout{1} = state.G;
varargout{2} = state.gbestval;
varargout{3} = exitflag;

output.problemtype	 = problemtype;
output.generations	 = state.generation;
output.funccount	 = state.generation*options.PopulationSize;
output.message		 = msg;
output.maxconstraint = max(max(g));

varargout{4} = output;
% END OF MAIN______________________________________________________________




% SUB-FUCTIONS_____________________________________________________________
function varargout = psooptimstruct(useValues)
% PSOOPTIMSTRUCT	De-struct problem structure

psostruct = struct('fitnessfcn',[],'nvars',[],'Aineq',[],'bineq',[],...
	'Aeq',[],'beq',[],'lb',[],'ub',[],'nonlcon',[]);

% Copy the values from the struct 'useValues' to 'probStruct'.
if ~isempty(useValues)
	copyfields = fieldnames(psostruct);
	Index = ismember(copyfields,fieldnames(useValues));
	for i = 1:length(Index)
		if Index(i)
			psostruct.(copyfields{i}) = useValues.(copyfields{i});
		end
	end
end

for i = 1:length(copyfields)
	varargout{i} = psostruct.(copyfields{i}); %#ok<AGROW>
end




function [pos,vel] = psogenseed(nvars,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options)
% PSOGENSEED	Generate random initial seed within all constraints

% Check if user-specified initial population violate constraints
if ~isempty(options.InitialPopulation)
	flag = psopenalize(...
		options.InitialPopulation,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options);
	
	if any(flag)
		warning('optim:pso:BadInitPop',['At least one particle in '...
			'OPTIONS ''InitialPopulation'' violate one or more constraints.'])
	end
end

% Number of non-user specified initial positions (initialize 10 x more than
% necessary)
N = 10*(options.PopulationSize-size(options.InitialPopulation,1));

% Set bounds of initial population based on lb, ub, InitRange and Seed
if ~isempty(options.PopInitRange)
	if size(options.PopInitRange,1)~=2 || size(options.PopInitRange,2)>nvars ||...
			any(options.PopInitRange(1,:)>options.PopInitRange(2,:))
		error('optim:pso:BadInitRange',...
			'The OPTION ''PopInitRange'' is incorrectly defined.')
	end
	% Number of intervals specified by user
	sze = size(options.PopInitRange,2);
	% Over-write lb and ub
	lb(1:sze,1) = options.PopInitRange(1,1:sze);
	ub(1:sze,1) = options.PopInitRange(2,1:sze);
end

% If lb not fully populated, initialize in invertval [-1 1]
lbsze = size(lb,1);
if lbsze<nvars
	lb(end+1:nvars,1) = 0;
end
% If lb not fully populated
ubsze = size(ub,1);
if ubsze<nvars
	ub(end+1:nvars,1) = 1;
end

% Set initial bounds to [0,1] if defined as [-inf,inf] by user.
lb(isinf(lb)) = 0; 
ub(isinf(ub)) = 1;

% Generate seed (N x nvars) within bounds lb and ub
pos = rand(N,nvars).*(repmat(ub',N,1) - repmat(lb',N,1)) + repmat(lb',N,1);

% Number of non-user specified initial velocities
V = options.PopulationSize-size(options.InitialVelocities,1);

if isempty(options.VelocityLimit)
	% Generate random initial velocities for all seeds
	vel = (-1 + 2*rand(V,nvars)).*(repmat(ub',V,1) - repmat(lb',V,1));
else
	% Shrink velocities if limits are specified
	vel = (-1 + 2*rand(V,nvars)).* repmat(options.VelocityLimit,V,1);
end
% Concatenate user and non-user specified initial velocities
vel = [options.InitialVelocities; vel];

LocalOptions = optimset('Simplex','off',...
	'LargeScale','off',...
	'Algorithm','active-set',...
	'Display','off') ;

% linprog fails if arguments are empty
if isempty(Aineq)
	Aineq = zeros(nvars);
	if isempty(bineq)
		bineq = zeros(nvars,1);
	end
end

isfeasible = [];

exitflag = zeros(N,1);

h = waitbar(0,['Generating initial population (0/' num2str(N/10) '). Please wait...']);
for n = 1:size(pos,1)
	if isempty(nonlcon)
		[pos(n,:), foo, exitflag(n)] =...
			linprog([],Aineq,bineq,Aeq,beq,lb,ub,pos(n,:),LocalOptions); %#ok<*ASGLU>
	else
		[pos(n,:), foo, exitflag(n)] =...
			fmincon(@void,pos(n,:),Aineq,bineq,Aeq,beq,lb,ub,nonlcon,LocalOptions);
	end

	% Set flag on feasible initial positions
	isfeasible = bitor(eq(exitflag, 1),eq(exitflag, -7));
	
	% Update waitbar
	waitbar(sum(isfeasible)/(N/10),h,['Generating initial population ('...
		num2str(sum(isfeasible)) '/' num2str(N/10) '). Please wait...']);
	
	% Stop seaching if necessary number of feasible initial positions is found
	if eq( sum(isfeasible), N/10 )
		break
	end
end

% Concatenate user and non-user specified seeds
pos = [options.InitialPopulation; pos(isfeasible,:)];

% Check if sufficient number of initial positions was found
if ~eq(size(pos,1), options.PopulationSize)
	error('pso:psogenseed:nosol',...
		'Failed to initialize swarm. Problem may be over-constrained.')
end

% Close waitbar
close(h)







function [state,exitflag] = psonearest(...
	nvars,state,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options)
% PSONEAREST	Diverge particles at boundary

LocalOptions = optimset('Simplex','off',...
	'LargeScale','off',...
	'Algorithm','active-set',...
	'Display','off',...
	'TolCon',options.TolCon) ;

% linprog fails if arguments are empty
if isempty(Aineq)
	Aineq = zeros(nvars);
	if isempty(bineq)
		bineq = zeros(nvars,1);
	end
end

pos = nan(options.PopulationSize,nvars);

exitflag = ones(options.PopulationSize,1);
for n = 1:options.PopulationSize
	if isempty(nonlcon)
		[pos(n,:), foo, exitflag(n,1)] = ...
			linprog([],Aineq,bineq,Aeq,beq,lb,ub,state.pos(n,:),LocalOptions);
	else
		[pos(n,:), foo, exitflag(n,1)] = ...
			fmincon(@void,state.pos(n,:),Aineq,bineq,Aeq,beq,lb,ub,nonlcon,LocalOptions);
	end
end

% Adjust velocity
state.vel = pos-(state.pos-state.vel);
% Absorb position
state.pos = pos;



function y = void(varargin)
% VOID	Dummy function for fmincon
y = 0;



function out = psodistjob(fitnessfcn,pos,options)
% PSODISTJOB	Evaluate fitness function

if strcmpi(options.Vectorized,'on')
	out = fitnessfcn(pos);
elseif strcmpi(options.Vectorized,'off')
	out = nan(options.PopulationSize,1);
	for n = 1:options.PopulationSize;
		out(n,1) =  fitnessfcn(pos(n,:));
	end
end



function [flag,g] = psopenalize(pos,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options)
% PSOPENALZE	Evaluate constraints

nvars = size(pos,2);

% If lb not fully populated, set to -inf
lbsze = size(lb,1);
if lbsze<nvars
	lb(end+1:nvars,1) = -inf;
end
% If ub not fully populated, set to inf
ubsze = size(ub,1);
if ubsze<nvars
	ub(end+1:nvars,1) = inf;
end

ilb = size(lb,1);
iub = ilb + size(ub,1);
iin = iub + size(Aineq,1);
ieq = iin + size(Aeq,1);

if ~isempty(nonlcon)
	[ctest,ceqtest] = nonlcon(pos(1,:));
	ctest = ctest(:); ceqtest = ceqtest(:);
	inl = ieq + size([ctest;ceqtest],1);
else
	inl = ieq;
end

itot = inl;

g = zeros(size(pos,1),itot) ;

for i = 1:size(pos,1)
	if ~isempty(lb) % Check lower bound
		g(i,1:ilb) = lb'-pos(i,:);
	end
	
	if ~isempty(ub) % Check upper bound
		g(i,ilb+1:iub) = pos(i,:)-ub';
	end
	
	if ~isempty(Aineq) % Check linear inequalities
		g(i,iub+1:iin) = Aineq*pos(i,:)'-bineq;
	end
	
	if ~isempty(Aeq) % Check linear equalities
		g(i,iin+1:ieq) = abs(Aeq*pos(i,:)'-beq);
	end
	
	if ~isempty(nonlcon) % Nonlinear constraint check
		[c,ceq] = nonlcon(pos(i,:));
		g(i,ieq+1:itot) = [c(:), abs(ceq(:))] ;
	end
end

% Flag particles violating at least one constraint
flag = max(g,[],2)>options.TolCon;




function state = psooutput(options,state,flag)
% PSOPUTPUT		Call plot and output functions

% Plot functions
if ~isempty(options.PlotFcns)
	functions = options.PlotFcns;
	if ~iscell(functions)
		functions = {functions};
	end
	for i = 1:length(functions)
		state = feval(functions{i},options,state,flag); drawnow
	end
end

% Output functions
if ~isempty(options.OutputFcns)
	functions = options.OutputFcns;
	if ~iscell(functions)
		functions = {functions};
	end
	for i = 1:length(functions)
		state = feval(functions{i},options,state,flag);
	end
end

if ~isfield(state,'StopFlag')
	state.StopFlag = false;
end


% Boundary method 'absorb'
function state = psoabsorb(state,A,b,Aeq,beq,lb,ub,nonlcon,options)

% Evaluate constraints
flag = psopenalize(state.pos,A,b,Aeq,beq,lb,ub,nonlcon,options);

ind = find(flag);

% Lower bound
lo = state.pos-state.vel;
% Upper bound
up = state.pos;

% Maximum number of bisections
MaxIter = 50;

for i = ind'
	for j = 1:MaxIter
		% New midpoint
		mid = (lo(i,:)+up(i,:))./2;
		
		% Evaluate constraints
		[foo, g] = psopenalize(mid,A,b,Aeq,beq,lb,ub,nonlcon,options);
		
		% Maximum violation of midpoint
		vmax = max(g,[],2);
		
		if vmax < -options.TolCon;
			lo(i,:) = mid;
		elseif vmax > options.TolCon;
			up(i,:) = mid;
		else % Converted to a solution
			break
		end
		
		% Span between lower and upper points
		span = norm(up(i,:)-lo(i,:))./2;
		
		if abs(span)<options.TolCon
			break
		end
	end
	
	% Update position
	state.pos(i,:) = mid;
	% Update velocity
	state.vel(i,:) = mid-lo(i,:);
end






