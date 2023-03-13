# mlp_cycleGAN

下载visdom
```
pip3 install visdom
```
```
python -m visdom.server
```
然后另起一个terminal->

Train

```python
python train.py --dataroot datasets/1001/ --input_nc 1 --output_nc 1
```

```python
python train.py --dataroot datasets/1001/ --input_nc 1 --output_nc 1
```

Test

```python
python test.py --dataroot datasets/1001/ --input_nc 1 --output_nc 1
```

```python
python test.py --dataroot datasets/1001/ --input_nc 1 --output_nc 1
```

改的地方用‘# 改’标注

源代码来自：
https://github.com/ZC119/Handwritten-CycleGAN

###################################################

https://console.cloud.google.com/compute/instances?project=s2313331-mlpractical&cloudshell=true

start instance

train：58行和110行

自己的terminal：
```python
conda activate mlp
gcloud auth login
gcloud compute ssh --zone "us-west1-b" "mlpractical-1"  --project "s2313331-mlpractical"
cd mlp_cw4
```

###################################################

计算fid：
```python
python -m fid2 path/real/folder path/output/folder
```
如：
```python
python -m calculate_fid datasets/1001/ output/images/
```
计算lpips:
```python
python calculate_lpips.py --real_path datasets/1001/ --fake_path datasets/images/
```

gnt转png：
0.1是train和test的比例
```python
python gnt2png.py datasets_gnt/HW/1001-f.gnt datasets_gnt/HW/SimHei.ttf 0.1 0.1
```