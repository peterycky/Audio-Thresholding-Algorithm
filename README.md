# Audio-Thresholding-Algorithm
Audio thresholding algorithm implemented in Python

This algorithm is performing action of __thresholding__ data which is different from filtration or correction filters. In short: 
* filtering scales amplitude accordingly to some given coefficients
* thresholding shows dynamically how we perceive some frequencies in binary domain (they are, or they aren't) 


## Installation
Make sure, you are using __Python 3__
To start fiddling with this script, install dependencies. Try using this command:

`python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose pydub easing-functions pylab`

## Getting ffmpeg set up

You may use **libav or ffmpeg**.

Mac (using [homebrew](http://brew.sh)):

```bash
# libav
brew install libav --with-libvorbis --with-sdl --with-theora

####    OR    #####

# ffmpeg
brew install ffmpeg --with-libvorbis --with-sdl2 --with-theora
```

Linux (using aptitude):

```bash
# libav
apt-get install libav-tools libavcodec-extra

####    OR    #####

# ffmpeg
apt-get install ffmpeg libavcodec-extra
```

Windows:

1. Download and extract libav from [Windows binaries provided here](http://builds.libav.org/windows/).
2. Add the libav `/bin` folder to your PATH envvar
3. `pip install pydub`

credit: [https://github.com/jiaaro/pydub#getting-ffmpeg-set-up]

## Settings
From line __14__ to line __22__ there are initial variables which, for now, serve as a setup for algorithm.

### Input song
On line __15__, there is definition of __input_song__:
`input_song = 'test-config.mp3'`
To change the song, simply change the string in single quotes. Relative paths should also work.

Output name is based on current date and time and for now, there is no possibility to change it, other than physically changing the code. For the curious: __line 635__

### Gain
Definition of Gain is on line __16__.
`gain = +0.0`
For more info, go to:
[https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentapply_gaingain]

### Demo
Be default demo is off. Turning demo ON will result in slower simulation execution and showing plots along the way.
Demo setting is on line __17__.
`demo_mode = False`

### Ear loss
Ear loss is measured hearing threshold for frequencies specified in FILTER_CENTER.
Ear loss is given in a logarythmic scale, where 0 is perfect hearing and 100 is complete deaf.

Initial setting simulates violinist's defficiency on left channel and old deafness on right channel.

Settings can be found on lines __21__ and __22__.

`left_ear_loss = [10, 20, 80, 50, 15, 10, 0, 0]`

`right_ear_loss = [0, 0, 0, 20, 40, 60, 75, 100]`
