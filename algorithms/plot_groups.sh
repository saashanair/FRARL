LOGDIR=CarSim-Python-v1-ppo2_rlf
GROUPDIR=groups
GROUPPATH=./logs/$LOGDIR/$GROUPDIR
rm -rf $GROUPPATH/eval
mkdir $GROUPPATH/eval

for STALIRO in no_staliro multiple_staliro
do
	for i in {1..10}
	do	    
	    OLD_PATH=$GROUPPATH/${STALIRO}-$i
	    NEW_PATH=$GROUPPATH/eval/${STALIRO}-$i
	    mkdir $NEW_PATH
	    for MONITOR in {10..19}
	    do
	        cp $OLD_PATH/0.$MONITOR.monitor.csv $NEW_PATH/.
	    done
	    cp $OLD_PATH/progress.csv $NEW_PATH/.
    done
done

python plot_groups.py $GROUPPATH/eval $GROUPPATH/evaluation.png

rm -rf $GROUPPATH/eval