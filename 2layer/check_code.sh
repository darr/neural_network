source ~/virtual_env/py2env/bin/activate
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf forward_neural_network.py
pylint --rcfile=pylint.conf cifar10_dataset.py
pylint --rcfile=pylint.conf graph.py
pylint --rcfile=pylint.conf func.py
deactivate

source ~/virtual_env/py3env/bin/activate
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf forward_neural_network.py
pylint --rcfile=pylint.conf cifar10_dataset.py
pylint --rcfile=pylint.conf graph.py
pylint --rcfile=pylint.conf func.py
deactivate
