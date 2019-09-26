![](examples/bk_video_B3.gif)
# Lighting-Swap-Cyclegan
This network is based on CycleGan(https://arxiv.org/pdf/1703.10593.pdf), with a couple small modifications to make it easy to swap lighting between sets of photos. This is the same network from my blog post: http://neuralvfx.com/lighting/lighting-swap-cyclegan/

My datasets were created by photographing the same site at two different times of day. The "Wat Mai Amataros" dataset can be downloaded here: http://neuralvfx.com/datasets/light_swap/wat_mai_amataros.rar

# Code Usage

Usage instructions found here: [user manual page](USAGE.md).


# Example Data Sets
## Wat Mai Amataros
### Sunny
![](examples/bk_setA.png)
### Cloudy
![](examples/bk_setB.png)
## Wat Choum Khong
### Cloudy
![](examples/set_A1.png)
### Sunny
![](examples/set_B1.png)

# Example Results
## Wat Mai Amataros
### Cloudy to Sunny 
#### (1: Real Cloudy Image — 2: Generated Sunny Image — 3: Real Sunny Image)
![](examples/bankok_pred_A5.png)
### Sunny to Cloudy
#### (1: Real Sunny Image — 2: Generated Cloudy Image — 3: Real Cloudy Image)
![](examples/bankok_pred_B3.png)
## Wat Choum Khong
### Cloudy to Sunny 
#### (1: Real Cloudy Image — 2: Generated Sunny Image — 3: Real Sunny Image)
![](examples/luang_pred_A12.png)
### Sunny to Cloudy 
#### (1: Real Sunny Image — 2: Generated Cloudy Image — 3: Real Cloudy Image)
![](examples/luang_pred_B13.png)

# Video Examples
## Wat Mai Amataros
### Cloudy to Sunny
![](examples/bk_video_A2.gif)
### Sunny to Cloudy
![](examples/bk_video_B3.gif)
## Wat Choum Khong
### Cloudy to Sunny
![](examples/luang_video_C.gif)
### Sunny to Cloudy
![](examples/luang_video_A.gif)
