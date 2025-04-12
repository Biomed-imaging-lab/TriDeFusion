install:
	conda env create -f environment.yml
    conda activate tridefusion

test:
	python -m pytest -vvv --cov=hello --cov=greeting \
		--cov=smath --cov=web tests
	python -m pytest --nbval notebook.ipynb	#tests our jupyter notebook
	#python -m pytest -v tests/test_web.py #if you just want to test web

deploy:
	docker build -t tridefusion .
	docker run -it tridefusion

test:
	python -m pytest -vvv --cov=hello --cov=greeting \
		--cov=smath --cov=web tests
	python -m pytest --nbval notebook.ipynb	#tests our jupyter notebook
	#python -m pytest -v tests/test_web.py #if you just want to test web

train:
	python inference.py --config train_config.yml

inference:
	python inference.py --noisy_img ./test_images/input.tif --output ./test_images/output.tif

debug:
	python -m pytest -vv --pdb	#Debugger is invoked

one-test:
	python -m pytest -vv tests/test_greeting.py::test_my_name4

debugthree:
	#not working the way I expect
	python -m pytest -vv --pdb --maxfail=4  # drop to PDB for first three failures

format:
	black *.py

lint:
	pylint --disable=R,C *.py

all: install lint test format