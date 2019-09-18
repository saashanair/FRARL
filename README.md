# Falsification-Based Robust Adversarial Reinforcement Learning

## Installation

1. Create conda environment **frarl** with necesssary packages.

	```
	conda env create -f environment.yml
	conda activate frarl
	pip install git+https://github.com/openai/baselines.git
	```


2. Download and install [MATLAB](https://www.mathworks.com/products/matlab.html)


3. Install MATLAB engine for Python


	* Open MATLAB and run following commands in MATLAB shell:


	```
	cd (fullfile(matlabroot,'extern','engines','python'))
	pwd
	```


	* In terminal, navigate to path of the matlab engine acquired in the previous step


	```
	cd <path_of_matlab_engine>
	```


	* Check path to created conda environment 

	```
	conda info -- env
	```

	* Install MATLAB engine in conda environment


	```
	conda activate frarl
	python setup.py build --build-base="<path_to_home_directory>" install --prefix="<path_to_conda_env>"
	```

	using <path_to_conda_env> as the build-base results in an error to copy the file, with the message that the name of the file is too long


4. Download MATLAB dependencies


	* Return to frarl directory

	```
	mkdir staliro_imports && cd staliro_imports
	```


	* Download S-Taliro [here](https://app.assembla.com/spaces/s-taliro_public/subversion/source/HEAD/trunk)
	* Download MatlabBGL [here](https://dgleich.github.io/matlab-bgl/)
	* Download Core_py2matlab [here](https://www.mathworks.com/matlabcentral/fileexchange/53717-core_py2matlab-convert-data-from-python-to-matlab-and-back)


	* Test if MATLAB engine install successfully and install S-Taliro


	```
	python
	import matlab.engine as me
	eng = me.start_matlab()
	eng.cd('trunk/')
	eng.setup_staliro(nargout=0)
	```


5. Extract HighD data

	* Request HighD dataset from [here](https://www.highd-dataset.com/) and replace the path to dataset in `/algorithms/highD/get_highd_data.py`

	```
	python ./algorithms/highD/get_highd_data.py
	```

6. Install simulator

	```
	cd simulator
	pip install -e .
	```

	* Test

	```
	python ./gym_car_acc/test/test_car_sim.py
	```


7. Run script to reproduce our results 

	```
	cd algorithms
	./run_ppo2_group.sh
	```