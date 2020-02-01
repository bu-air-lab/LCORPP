## Prerequisite
Download P-Log [PLOG](http://www.depts.ttu.edu/cs/research/documents/plog-1.0.0.tar.gz) 
```sh
$ tar xvfz plog-1.0.0
$ cd plog-1.0.0/plog/src
$ sudo cmake .
$ sudo make install
```
Download [smodels](http://www.tcs.hut.fi/Software/smodels/src/smodels-2.34.tar.gz)
```sh
$ tar xvfz smodels-2.34.tar.gz 
$ cd smodels-2.34
$ make
```
Download [lparse](http://www.tcs.hut.fi/Software/smodels/src/lparse-1.1.2.tar.gz)

```sh
$ tar xvfz lparse-1.1.2
$ cd lparse-1.1.2
$ ./configure
$ make
$ make install
```

## Dependencies

```sh
$ sudo apt-install -y cmake
$ sudo apt-get install -y bison 
$ sudo pip install virtualenv      
$ virtualenv .env                  
$ source .env/bin/activate         
$ pip install -r requirements.txt  
```