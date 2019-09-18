Guide that tracks the problems that were encountered while working with S-Taliro, esp in combination with Tensorflow, and solutions to them.

1. "max_dp_taliro" not found
	Sol: run setup_dp_taliro.m that can be found under stalrio/dp_taliro

2. Importing Tensorflow throws CXXABI error
	Sol: export LD_PRELOAD = /usr/lib/x86_64_linux_gui/libstdc++.so.6:$LD_PRELOAD

3. Importing Tensorflow throws....
	Sol: export LYLD_LIBRARY_PATH = /home/nairs/.conda/env/hiwi/lib/:$DYLD_LIBRARY_PATH

4 Version mismatch error with HDFS due to Matlab and Python using different versions
	Sol: conda uninstall h5py --> use JSON to save and load weights insteads with the methods model.get_weights() and model.set_weights()