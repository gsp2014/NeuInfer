# NeuInfer: Knowledge Inference on N-ary Facts

This project provides the tensorflow implementation of the knowledge inference model NeuInfer on n-ary facts, published in ACL'20.

## Usage
### Prerequisites
- Python 3.6
- Tensorflow 1.4.0

### Prepare data
Transform the representation form of facts for [JF17K](https://github.com/lijp12/SIR). Convert each attribute value sequence of a specific n-ary relation to a primary triple coupled with a set of its auxiliary description(s):

    python JF17K2hrtav.py

Transform the representation form of facts for [WikiPeople](https://github.com/gsp2014/WikiPeople). Convert each set of attribute-value pairs to a primary triple coupled with a set of its auxiliary description(s):

    python WikiPeople2hrtav.py

Build data before training and test for JF17K and WikiPeople:

    python builddata.py --sub_dir JF17K_version1 --dataset_name JF17K_version1
    python builddata.py --sub_dir WikiPeople --dataset_name WikiPeople

Build data for filtering the right facts in negative sampling or computing the filtered metrics when evaluation:

    python builddata.py --sub_dir JF17K_version1 --dataset_name JF17K_version1 --if_permutate True --bin_postfix _permutate
    python builddata.py --sub_dir WikiPeople --dataset_name WikiPeople --if_permutate True --bin_postfix _permutate

### Training
To train NeuInfer:

    python train.py --sub_dir JF17K_version1 --dataset_name JF17K_version1 --wholeset_name JF17K_version1_permutate --model_name JF17K_version1_opt --embedding_dim 100 --hrtFCNs_layers 2 --hrtavFCNs_layers 1 --g_theta_dim 1200 --weight 0.1 --batch_size 128 --learning_rate 0.00005 --n_epochs 5000 --saveStep 100
    python train.py --sub_dir WikiPeople --dataset_name WikiPeople --wholeset_name WikiPeople_permutate --model_name WikiPeople_opt --embedding_dim 100 --hrtFCNs_layers 1 --hrtavFCNs_layers 1 --g_theta_dim 1000 --weight 0.3 --batch_size 128 --learning_rate 0.0001 --n_epochs 5000 --saveStep 100
            
### Evaluation
Files `eval.py` and `eval_bi-n.py` provide four evaluation metrics, including the Mean Reciprocal Rank (MRR), Hits@1, Hits@3, and Hits@10 in filtered setting. In these two files, parameter **--valid_or_test** indicates whether to evaluate NeuInfer in the validation set (set to 1) or test set (set to 2).

To evaluate NeuInfer in the validation set (JF17K lacks a validation set):

    python eval.py --sub_dir WikiPeople --dataset_name WikiPeople --wholeset_name WikiPeople_permutate --model_name WikiPeople_opt --embedding_dim 100 --hrtFCNs_layers 1 --hrtavFCNs_layers 1 --g_theta_dim 1000 --weight 0.3 --batch_size 128 --n_epochs 5000 --start_epoch 100 --evalStep 100 --valid_or_test 1 --gpu_ids 0,1,2,3

To evaluate NeuInfer in the test set:

    python eval.py --sub_dir JF17K_version1 --dataset_name JF17K_version1 --wholeset_name JF17K_version1_permutate --model_name JF17K_version1_opt --embedding_dim 100 --hrtFCNs_layers 2 --hrtavFCNs_layers 1 --g_theta_dim 1200 --weight 0.1 --batch_size 128 --n_epochs 5000 --start_epoch 100 --evalStep 100 --valid_or_test 2 --gpu_ids 0,1,2,3
    python eval.py --sub_dir WikiPeople --dataset_name WikiPeople --wholeset_name WikiPeople_permutate --model_name WikiPeople_opt --embedding_dim 100 --hrtFCNs_layers 1 --hrtavFCNs_layers 1 --g_theta_dim 1000 --weight 0.3 --batch_size 128 --n_epochs 5000 --start_epoch 100 --evalStep 100 --valid_or_test 2 --gpu_ids 0,1,2,3

File `eval_bi-n.py` provides more detailed results on binary and n-ary categories. It is used in the same way as `eval.py`.

Note that, it takes a lot of time to evaluate NeuInfer, since we need to compute a score via NeuInfer for each candidate (each element in the set of relations and attributes/entities and attribute values). To speed up the evaluation process, `eval.py` and `eval_bi-n.py` are implemented in a multi-process manner.

## Citation
If you found this codebase or our work useful please cite:

    @inproceedings{NeuInfer,
      title={NeuInfer: Knowledge inference on n-ary facts},
      author={Guan, Saiping and Jin, Xiaolong and Guo, Jiafeng and Wang, Yuanzhuo and Cheng, Xueqi},
      booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL'20)},
      year={2020},
      pages={6141--6151}
    }

## Related work
[Link Prediction on N-ary Relational Data](https://github.com/gsp2014/NaLP)
