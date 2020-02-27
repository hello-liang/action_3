function state = psoplotbestf(options,state,flag)
% PSOPLOTBESTF	 Simple plotting function for PSO. Based on Matlab function
% gaplotbestf
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
 
state.StopFlag = false;

% Detect and close older PSO figure if one is still around
fig = findobj(0,'type','figure','name','Particle Swarm Optimization');

switch flag
	case 'init'
% 		% Detect and close older PSO figure if one is still around
% 		fig = findobj(0,'type','figure','name','Particle Swarm Optimization');
		if ~isempty(fig)
			close(fig)
		else
			figure
		end
		
		% Create new figure
		hold on;
		
		% Prepare axes
		set(gca,'xlim',[state.generation, options.Generations]);
		xlabel('Generation','interp','none');
		ylabel('Fitness value','interp','none');
		
		% Plot
		plotBest = plot(state.generation,state.gbestval,'.k');
			set(plotBest,'Tag','psoplotbestf');
		plotMean = plot(state.generation,mean(state.pbestval),'.b');
			set(plotMean,'Tag','psoplotmean');
			
		title(['Best: ',' Mean: '],'interp','none')
		set(gcf,'Name','Particle Swarm Optimization',...
				'ToolBar','none','NumberTitle','off')
	case 'iter'
		figure(fig)
		
		best = state.gbestval;
		m    = mean(state.pbestval);
		
		plotBest = findobj(get(gca,'Children'),'Tag','psoplotbestf');
		plotMean = findobj(get(gca,'Children'),'Tag','psoplotmean');
		
		newX = [get(plotBest,'Xdata') state.generation];
		newY = [get(plotBest,'Ydata') best];
		
		set(plotBest,'Xdata',newX, 'Ydata',newY);
		newY = [get(plotMean,'Ydata') m];
		set(plotMean,'Xdata',newX, 'Ydata',newY);
		set(get(gca,'Title'),'String',...
			['Best: ',num2str(best),' Mean: ',num2str(m)]);
	case 'done'
		figure(fig)
		
		h = legend('Best','Mean');
		set(h,'FontSize',8);
		
		hold off;
end

% set(gcf,'HandleVisibility','off')
