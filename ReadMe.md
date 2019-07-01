# Pytorch Version speech enhancemnet in cldnn
## author: yxhu
### thanks to Ke Wang and awni's [repo](https://github.com/awni/speech/)


# update log:
### 2019-4-21
 add clip\_grad\_norm in step/run\_cldnn.py 
 fix memory leak bug in ./tools/dataset.py's collat\_fn:
            return numpy.array to torch.tensor, which can be lead to memory leak
 change mse in frames to mse in samples
### 20190630 
1. change the conv2d to like tf's 'same' padding 
2. add SRU: [SRU](https://github.com/taolei87/sru)
3. add 1k6h train strategy: warmup
4. add eval
