# BIRDNN: Behavior-Imitation Based Repair of Deep Neural Networks

**BIRDNN**(**B**ehavior-**I**mitation Based **R**epair of **DNN**s) supports alternative retraining style and fine-tuning style repair simultaneously. 

Code for the paper "BIRDNN: Behavior-Imitation Based Repair of Deep Neural Networks". 





## Acknowledgements

The code is based on [Veritex](https://github.com/Shaddadi/veritex) and [CARE](https://github.com/sunbingsmu/care). Thank to their wonderful and clear implementation.





## Installation and System Requirements

This tool has been confirmed to work and tested with only Python3.7.

1. Install BIRDNN pkg with pip.

   ```
   python -m pip install .
   ```

2. Set path to /BIRDNN under this repository.

   ```
   export PYTHONPATH='<YOUR_REPO_PATH>/BIRDNN'
   export OPENBLAS_NUM_THREADS=1
   export OMP_NUM_THREADS=1
   ```







## Run experiments

If you want to run DNN retraining code, here is an example of code running commands:

```
python BIRDNN/repair/retraining.py --network 19 --alpha 0.5 --beta 0.5 --lr 0.001 --repair_neurons_num 20 --epoch 200
```



If you want to run DNN fine-tuning code, here is an example of code running commands:

```
python BIRDNN/repair/fine_tuning.py --network 19 --alpha 0.5 --repair_neurons_num 20 --lr 0.001
```



If you want to output the results of fine-tuning single-layer running, here is an example of code running:

```
python BIRDNN/repair/fine_tuning_single_layer.py --network 19 --alpha 0.5 --repair_neurons_num 20 --lr 0.001
```







## If you find our paper/codes helpful, please cite:

```
@inproceedings{yang2022neural,
  title={Neural network repair with reachability analysis},
  author={Yang, Xiaodong and Yamaguchi, Tom and Tran, Hoang-Dung and Hoxha, Bardh and Johnson, Taylor T and Prokhorov, Danil},
  booktitle={Formal Modeling and Analysis of Timed Systems: 20th International Conference, FORMATS 2022, Warsaw, Poland, September 13--15, 2022, Proceedings},
  pages={221--236},
  year={2022},
  organization={Springer}
}
```



```
@inproceedings{sun2022causality,
  title={Causality-based neural network repair},
  author={Sun, Bing and Sun, Jun and Pham, Long H and Shi, Jie},
  booktitle={Proceedings of the 44th International Conference on Software Engineering},
  pages={338--349},
  year={2022}
}
```

