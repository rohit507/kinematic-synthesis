# -*- mode: ruby -*-
# vi: set ft=ruby :

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|

  # Our default provider is virtualbox, which allows us to do all of our build
  # work inside of a VM. The larger memory allocation mainly allows us to
  # build DReal without hitting swap. 
  config.vm.provider "virtualbox" do |vb,override|

    # Every Vagrant development environment requires a box. You can search for
    # boxes at https://vagrantcloucom/search.
    override.vm.box = "ubuntu/xenial64"
    
    # Display the VirtualBox GUI when booting the machine
    # vb.gui = true
    
    # Customize the amount of memory on the VM:
    vb.memory = 4096 * 4

    # Customize the number of CPUs on the VM:
    vb.cpus = 6
  end

  # LXC is the alternate (NOTE: currently broken) provider, that uses a
  # container instead of a VM. This tends to reduce overhead while providing
  # identical isolation as a VM, but ends up only really working on linux,
  # as you generally need to be hosting a sufficiently similar 
  config.vm.provider "lxc" do |lxc,override|

    # We're using a box that seems to be well managed, theoretically we could
    # just use this one box for both virtualbox and lxc, but I'd rather keep
    # the most stable looking one at any point. 
    override.vm.box = "magneticone/ubuntu"

    # We use a directory backing store here, because others seem to cause
    # issues with the latest versions of vagrant.
    lxc.backingstore = 'dir'

  end
  
  # Just common utilities that I like to have, nothing too special.  
  config.vm.provision "shell", inline: <<-SHELL
    echo "#### Installing Common Software ####"

    apt-get update
    apt-get upgrade -y 
    apt-get install -qq git subversion software-properties-common tree
    apt-get install -qq htop build-essential curl wget openssl graphviz
  SHELL

  # Take the initscript in our build directory and make sure that we
  # run it for every user login shell.
  #
  # The initscript does a number of things like adding DReal to the pythonpath
  # and ensuring that the right python version is defaulted.
  #
  # Go read it if you're manually installing the dependencies for this.
  config.vm.provision "shell", inline: <<-SHELL
    echo "#### Setting up Environment ####"

    ln -sf /vagrant/setup/init.sh /etc/profile.d/path-init.sh
  SHELL
  
  # Install python3, pip, pyenv and virtualenv.
  #
  # Note that we are install python 3.6.5, if you're performing a manual
  # install, you should use whatever method you're comfortable with to get
  # that version running.
  #
  # We install python 3.6 with a ppa so that we can later use it during the
  # build process for DReal's python bindings.
  config.vm.provision "shell", inline: <<-SHELL
    echo "#### Installing Python3 ####"

    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update
    apt-get install -qq python3.6 python3.6-venv libpython3.6-dev
    apt-get install -qq zlib1g-dev libncurses5-dev libdb5.3-dev libexpat1-dev
    apt-get install -qq liblzma-dev tk-dev
    apt-get install -qq libncursesw5-dev libreadline-dev libssl-dev
    apt-get install -qq libgdbm-dev libc6-dev libsqlite3-dev tk-dev libbz2-dev
    su vagrant -l -c 'curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash'
    su vagrant -l -c 'pyenv update'
    su vagrant -l -c 'pyenv install 3.6.5'
    su vagrant -l -c 'pyenv global 3.6.5'
  SHELL

  # build DReal as per https://github.com/dreal/dreal4
  # with python3 bindings as in https://github.com/dreal/dreal4/issues/69 albeit
  # while generating bindings specifically for python 3.6 rather than just the
  # default python3. 
  #
  # NOTE :: This will take a while ...
  config.vm.provision "shell", inline: <<-SHELL
    echo "#### Installing DReal ####"

    if [ ! -d "/opt/dreal/" ]
    then
      echo "LOG: DReal not found, building"
      rm -rf /install/DReal
      rm -rf /opt/dreal/
      mkdir /install/DReal -p
      cd /install/DReal
      git clone https://github.com/dreal/dreal4.git ./
      git checkout 5e7c80a -b fixed-install-version
      ./setup/ubuntu/$(lsb_release -r -s)/install_prereqs.sh
      patch tools/pybind11.BUILD.bazel /vagrant/setup/pybind11.patch
      patch dreal/workspace.bzl /vagrant/setup/workspace.patch
      bazel build //:archive
      tar -xvf bazel-bin/archive.tar.gz --directory=/
    else
      echo "LOG: DReal found, skipping build" 
    fi
  SHELL

  # git checkout da71100 -b fixed-install-version


end
