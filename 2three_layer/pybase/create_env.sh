#假设已经安装了vitralenv　并且环境中有Python2 和python3
rm -rf ~/.pybase_env
mkdir ~/.pybase_env
cd ~/.pybase_env
virtualenv -p /usr/bin/python2 py2env
source ~/.pybase_env/py2env/bin/activate
pip install Pillow
pip install tornado
pip install mysqlclient
deactivate
virtualenv -p /usr/bin/python3 py3env
source ~/.pybase_env/py3env/bin/activate
pip install Pillow
pip install tornado
#pip install mysqlclient
#3.5 现在还不支持MySQLdb
pip install PyMySQL
deactivate
