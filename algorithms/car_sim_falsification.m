function [r, h] = car_sim_falsification(seed)

% disp(' ')
model = @CarSimBlackbox;

% disp(' ')
% disp('The constraints on the initial conditions defined as a hypercube:')
init_cond = [10 600.; 1.45 64.77];

% disp(' ')
% disp('The constraints on the input signal defined as a range:')
input_range = [-10. 10.];

% disp(' ')
% disp('The number of control points for the input signal:')
cp_array = [1500];

% disp(' ')
% disp('The specification:')
phi = '[](!is_collision /\ !abs_ego_vel)';

ii = 1;
preds(ii).str='is_collision';
preds(ii).A = [-1 0 0 0 0];
preds(ii).b = 4;

ii = ii + 1;
preds(ii).str='abs_ego_vel';
preds(ii).A = [0 0 1 0 0];
preds(ii).b = 0;


% disp(' ')
% disp('Total Simulation time:')
time = cp_array(1);

% disp(' ')
% disp('Create an staliro_options object with the default options:')
opt = staliro_options();

% disp(' ')
% disp('Change options:')
opt.black_box = 1;
opt.interpolationtype = {'pchip'};
opt.taliro_metric = 'none';
opt.SampTime = 1;
opt.map2line = 0;
opt.seed = seed;

% set number of workers for parallel computation:
opt.n_workers = 1;
%opt.normalization = 1; % normalization throws an error: Reference to non-existent field 'Normalized'.
                                                        %
                                                        %Error in staliro (line 760)
                                                        %    if any(find([preds.Normalized]==0))

% disp(' ')
% disp (' Using Simulated Annealing ')
% disp (' ')
% solver_id = input ('Select an option (1-2): ')
opt.optimization_solver = 'CE_Taliro';
opt.runs = 1;
opt.optim_params.n_tests = 50;
opt.optim_params.num_iteration = 10;
opt.dispinfo = 0;
% opt

% opt.optim_params

% disp(' ')
% disp('Running S-TaLiRo ...')
tic
[results, history] = staliro(model,init_cond,input_range,cp_array,phi,preds,time,opt);
runtime=toc;

fprintf('runtime=%d\n', runtime)

% class(results)

% results.run

% size(results.run)

r = [];

h = [];

for n = 1:opt.runs
    robustness = results.run(n).bestRob;

	% only collect falsified samples
    if robustness < 0
        r = [r, results.run(n).bestSample];
	h = [h, robustness];
    end
end

size(r);

size(h);

results.run.bestRob;

% results.optRobIndex

% disp('results.run.bestSample')
results.run(results.optRobIndex).bestSample;
% disp('Size of results.run.bestSample')
% size(results.run(results.optRobIndex).bestSample);
% disp('history.samples')
history.samples;

% disp('check for falsification')
fprintf('falsification = %d\n', results.run(results.optRobIndex).falsified)

% disp('check best robustness value')
fprintf('best_robustness = %d\n', results.run(results.optRobIndex).bestRob)
% class(x)

% r = results.run(results.optRobIndex).bestSample;
% h = results.run(results.optRobIndex).bestRob;

end

