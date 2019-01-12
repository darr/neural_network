source ~/.pybase_env/py2env/bin/activate
python pytest.py
deactivate
source ~/.pybase_env/py3env/bin/activate
python pytest.py
deactivate
python pyclean.py
