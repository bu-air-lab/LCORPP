
## Downloaind the repository
```sh
cd ~
git clone git clone --branch aaai https://github.com/bu-air-lab/LCORPP.git
mkdir software       
```


## Dependencies

```sh
$ sudo apt-install -y cmake
$ sudo pip install virtualenv      
$ virtualenv -p python3 ~/.env                  
$ source ~/.env/bin/activate         
$ pip install -r LCORPP/requirements.txt  
```




## Prerequisite (Ubuntu 16.04)

Installing [smodels](http://www.tcs.hut.fi/Software/smodels/src/smodels-2.34.tar.gz)
```sh
$ cd ~/Software
$ wget http://www.tcs.hut.fi/Software/smodels/src/smodels-2.34.tar.gz 
$ tar xvfz smodels-2.34.tar.gz 
$ cd ~/Downloads/smodels-2.34
$ sudo make
$ sudo make install
```
Installing [lparse](http://www.tcs.hut.fi/Software/smodels/src/lparse-1.1.2.tar.gz)

```sh
$ cd ~/Software 
$ wget http://www.tcs.hut.fi/Software/smodels/src/lparse-1.1.2.tar.gz 
$ tar xvfz lparse-1.1.2
$ cd ~/Downloads/lparse-1.1.2
$ cp ~/LCORPP/library.cc ~/Downloads/lparse-1.1.2/src/
$ ./configure
$ make
$ make install
```

Installing P-Log [PLOG](http://www.depts.ttu.edu/cs/research/documents/plog-1.0.0.tar.gz) 
```sh
$ cd ~/Software 
$ wget http://www.depts.ttu.edu/cs/research/documents/plog-1.0.0.tar.gz 
$ tar xvfz plog-1.0.0
$ cd ~/Downloads/plog-1.0.0/plog/src
$ sudo cmake .
$ sudo make install
```

Installing Sarsop

```sh
$ cd ~/Software
$ git clone https://github.com/AdaCompNUS/sarsop.git
$ cd sarsop/src
$ make 
```

## Run the simulator
```sh
$ cd ~/LCORPP
$ python simulator.py  
```


