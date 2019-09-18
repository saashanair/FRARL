function [T, XT, YT, LT, CLG, GRD] = run_simulation_with_python(XPoint, simTime, steptime, InpSignal)

% Import Python modules:
%pyversion /anaconda3/envs/hiwi_pygame/bin/python -- to ensure that matlab
%uses the python from your specific env by default https://de.mathworks.com/matlabcentral/answers/325435-python-api-setup-environments-how-to-get-matlab-to-use-other-env-as-interpeter

size(XPoint);
XPoint;

size(InpSignal);

InpSignal;

size(simTime);

%% Run the simulation and receive the trajectory:

% import play script from baseliens
play_im = py.importlib.import_module('play');
traj = py.play.play_sim(XPoint', simTime, InpSignal');
% Convert trajectory to matlab array
mattraj = Core_py2matlab(traj);

% Depending on you trajectory data returned from python code, the following should be customized:
%YT = [];
LT = [];
CLG = [];
GRD = [];
if isempty(mattraj)
    T = [];
    XT = [];
else
    T = mattraj(:,1)/1000.0;
    XT = mattraj(:, 2:end);
    YT = XT;

end
end
