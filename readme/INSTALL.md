# Dependencies and Installation

There are a few major install methods that we use.
The default spins up a virtual machine for you.
The containerized method doesn't work yet. 
The manual method should be adaptable enough to let you figure out how to 
install the system however you want. 

### Default Method (Vagrant + VirtualBox) 

We will be using [Vagrant](https://www.vagrantup.com/) for setting up an 
isolated build environment to work in. This means that we can automate much 
of the setup process, and generally make sure that there are no conflicts with 
your existing environment setup. 

  - **Install Vagrant:** Download and install 
    [Vagrant](https://www.vagrantup.com/downloads.html).
    
We be using [VirtualBox](https://www.virtualbox.org) as a Vagrant "provider", 
basically an virtualization mechanism that Vagrant can use. 
Others might be supported in the future but this is the default. 

  - **Install VirtualBox:** Download and install 
    [VirtualBox](https://www.virtualbox.org/wiki/Downloads).

The container is setup according to the instructions in the 
(Vagrantfile)[Vagrantfile]. 
It will install python, DReal, and any other dependencies into the VM, as well
as mirror the root directory of this repository to `/vagrant/` in the VM. 

    $ cd <project-root> 
    $ vagrant up
    
Running this for the first time will download a number of files and set up your
virtual machine. 
Get a coffee or something, this will take a while. 
If you want to reset the virtual machine for some reason, just run 
`vagrant destroy` and then `vagrant up` again. 

Once that is done for the first time, you can enter the virtual machine with 
the following commands. 

    $ vagrant up
    $ vagrant ssh

in particular `vagrant ssh` will drop you into a terminal in your VM, where the
files you are working on 


### Containerized Method (Vagrant + LXC) 

In addition to the installs needed in the default method, you will also need 
[LXC](https://linuxcontainers.org/lxc/).
LXC allows you run your scripts inside a container that can avoid much of the 
overhead of a fully virtualized setup.

  - **Install LXC:** Install using the instructions 
    [here](https://linuxcontainers.org/lxc/getting-started/).
    
To allow vagrant to use this however, you also need the vagrant plugin
[vagrant-lxc](https://github.com/fgrehm/vagrant-lxc) which allows vagrant to
use LXC instead of virtualbox. 

  - **Install vagrant-lxc:** Follow the instructions 
    [here](https://github.com/fgrehm/vagrant-lxc)
    Usually this just means running the command 
    `vagrant plugin install vagrant-lxc`

Then you can setup the container much the same way as the VM, albeit with a 
different default provider. 

    $ cd <project-root> 
    $ vagrant up --provider=lxc

At the moment this has some issues with LXC 3.0+ so you might have to downgrade.
Alternately, by the time you read this vagrant-lxc might have updated. 

### Manual Install

The short version is just read the [Vagrantfile](../Vagrantfile) and figure out the
analogous commands for your system.
There should be enough comments and links for you to figure stuff out. 
