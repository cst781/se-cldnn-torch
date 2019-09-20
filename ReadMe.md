# Pytorch Version singel channel speech enhancemnet in cldnn
## author: yxhu
### thanks to Ke Wang and awni's [repo](https://github.com/awni/speech/)

# How to use it ?
## 1. install requirements 

### sru
```bash
git https://github.com/asappresearch/sru.git
cd sru 
pip install -r requirements.txt
python setup.py install
```
### pypesq
```bash
git clone https://github.com/vBaiCai/python-pesq.git
cd python-pesq
python setup.py install
```
### pystoi
```bash
pip install pystoi
```

## 2. download dataset 
### Aishell-1
https://www.openslr.org/33/
### MUSAN
https://www.openslr.org/17/

## 3. prepare train. dev and test data 
```bash
find ${aishell_dir}/train -iname "*.wav" > train.lst
find ${aishell_dir}/dev -iname "*.wav" > dev.lst
find ${aishell_dir}/test -iname "*.wav" > test.lst
find ${musan}/test -iname "*.wav" > noise.lst # Attention!!, please do not add musan/speech into noise.lst

bash tools/prepare_train.sh
bash tools/prepare_test.sh
```

## 4. train model
```bash
bash run_sruc.sh
```

## 5. eval model
```bash
bash eval.sh
```



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

### 20190920
1. update data prepare pipeline
2. add psa,psm 

