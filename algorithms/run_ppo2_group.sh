
LOGDIR=CarSim-Python-v1-ppo2_rlf
mkdir ./logs
mkdir ./logs/$LOGDIR
mkdir ./logs/$LOGDIR/groups

for STALIRO in no_staliro multiple_staliro
	do
	STALIRO_ROOT=./logs/$LOGDIR/${STALIRO}_groups
	# rm -rf $STALIRO_ROOT
	mkdir $STALIRO_ROOT

	for seed in {1..10}
	do	
		SEEDROOT=$STALIRO_ROOT/${STALIRO}-$seed
		rm -rf ./logs/$LOGDIR/$STALIRO/
		rm staliro_results_*.npy
		python run.py --alg=ppo2_rlf --env=CarSim-Python-v1 --num_timesteps=1e6 --nsteps=256 --num_env=10 --save_interval 10 \
	 	--call_staliro $STALIRO --play True --seed $seed

		if [[ "$STALIRO" != "no"* ]]; then
			mkdir ./logs/$LOGDIR/$STALIRO/staliro_results
			mv staliro_results_*.npy ./logs/$LOGDIR/$STALIRO/staliro_results/.
		fi
		mv ./logs/$LOGDIR/$STALIRO/ $SEEDROOT
		cp -r $SEEDROOT ./logs/$LOGDIR/groups/${STALIRO}-$seed

	done
done

rm ~/matlab_crash_dump.*
./plot_groups.sh
