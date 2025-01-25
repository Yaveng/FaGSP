Hi there👋.

This is the official implementation of Frequency-aware Graph Signal Processing for Collaborative Filtering, a.k.a. FaGSP, which is accepted by The Web Conference 2025 (Short Paper). For Completed version, please refer to [arXiv](https://arxiv.org/abs/2402.08426).

We hope this code helps you well. If you use this code in your work, please cite our paper.

```
Frequency-aware Graph Signal Processing for Collaborative Filtering
Jiafeng Xia, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Li Shang and Ning Gu.
The Web Conference (WWW). 2025
```

#### How to run this code

##### Step 1: Check the compatibility of your running environment. 

Generally, different running environments will still have a chance to cause different experimental results though all random processes are fixed in the code. Our running environment is 

```
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
- GPU: Tesla T4
- Memory: 251.5G
- Operating System: Ubuntu 16.04.7 LTS (GNU/Linux 4.15.0-142-generic x86_64)
- CUDA Version: 10.0
- Python packages:
  - numpy==1.19.5
  - pandas==1.1.5
  - python=3.6.13
  - scikit-learn==0.22
  - scipy==1.5.4
  - pytorch==1.8.0
```

##### Step 2: Prepare the datasets. 

Please put your datasets under the directory `dataset/XXX/`, where ```xxx``` is the name of the dataset, e.g., ```ml100k```. If the directory doesn't exist, please create it first. Please note that if you use your own datasets, check their format so as to make sure that it matches the input format of `FaGSP`. 


##### Step 3: Run the code.

* For ```LastFM``` dataset:
  ```python
  python main.py --dataset lastfm --pri_factor1 64 --pri_factor2 64 --alpha1 0.85 --alpha2 0.35 --order1 14 --order2 14 --q 0.8
  ```

* For ```ML1M``` dataset:
  ```python
  python main.py --dataset ml1m --pri_factor1 256 --pri_factor2 128 --alpha1 0.3 --alpha2 0.5 --order1 12 --order2 14 --q 0.7
  ```

