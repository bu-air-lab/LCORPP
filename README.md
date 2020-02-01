
## Downloaind the repository
```sh
cd ~
git clone git clone --branch aaai https://github.com/bu-air-lab/LSTM-CORPP.git
```


## Dependencies

```sh
$ sudo apt-install -y cmake
$ 
$ sudo pip install virtualenv      
$ virtualenv .env                  
$ source .env/bin/activate         
$ pip install -r LSTM-CORPP/requirements.txt  
```




## Prerequisite (Ubuntu 16.04)

Download [smodels](http://www.tcs.hut.fi/Software/smodels/src/smodels-2.34.tar.gz)
```sh
$ tar xvfz smodels-2.34.tar.gz 
$ cd ~/Downloads/smodels-2.34
$ sudo make
$ sudo make install
```
Download [lparse](http://www.tcs.hut.fi/Software/smodels/src/lparse-1.1.2.tar.gz)

```sh
$ tar xvfz lparse-1.1.2
$ cd ~/Downloads/lparse-1.1.2
$ cp ~/LSTM-CORPP/library.cc ~/Downloads/lparse-1.1.2/src/
$ ./configure
$ make
$ make install
```

Download P-Log [PLOG](http://www.depts.ttu.edu/cs/research/documents/plog-1.0.0.tar.gz) 
```sh
$ tar xvfz plog-1.0.0
$ cd ~/Downloads/plog-1.0.0/plog/src
$ sudo cmake .
$ sudo make install
```
