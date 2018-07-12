#!/bin/sh

# Add DReal's python bindings to the pythonpath. 
export PYTHONPATH=/opt/dreal/4.18.03.3/lib/python2.7/site-packages:${PYTHONPATH}

# And the binary directory to the right place 
export PATH=/opt/dreal/4.18.03.3/bin:${PATH}

# add the library to the library search path
export LD_LIBRARY_PATH=/opt/dreal/4.18.03.3/lib${LD_LIBRARY_PATH}

# Add pyenv to the path, and initialize it on each startup
export PATH=/home/vagrant/.pyenv/bin:$PATH
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Make sure that when you login, you move to the directory where
# the project folder is mounted. 
cd /vagrant/

# Make sure that all the packages we need are installed
pip install -r requirements.txt
