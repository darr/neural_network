env_path=~/.three_layer
source $env_path/py2env/bin/activate
python app_test.py
deactivate

source $env_path/py3env/bin/activate
python app_test.py
deactivate
