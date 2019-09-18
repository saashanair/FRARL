### Rules for logging experiment results:
+ logging path = root / env + algo / experiment
+ Root dir is hard coded in constants.LOG_DIR
+ Env, algo, experiment are extracted from args and extra_args
+ In addition, algs and alg_kwargs are saved both in experiment folder (for logging purpose)and root folder (for play_sim function)



### To load from model and continue experiments
+ Copy paste `args.npz` and `alg_kwargs.npz` from experiment folder to root folder (./logs)
+ Use --load_path True in args

### Experiment option

+ --call_staliro has the following values: [no_staliro, multiple_staliro, fixed_staliro, randomized_staliro, weighted_queue, variable_start, success_counter]
