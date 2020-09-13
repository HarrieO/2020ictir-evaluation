# Taking the Counterfactual Online: Efficient and Unbiased Online Evaluation for Ranking
This repository contains the code used for the experiments in "Taking the Counterfactual Online: Efficient and Unbiased Online Evaluation for Ranking" published at ICTIR 2020 ([preprint available](https://arxiv.org/abs/2007.12719)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our ICTIR 2020 paper:

```
@inproceedings{oosterhuis2020taking,
  title={Taking the Counterfactual Online: Efficient and Unbiased Online Evaluation for Ranking},
  author={Harrie Oosterhuis and Maarten de Rijke},
  year={2020},
  booktitle={Proceedings of the 2020 International Conference on The Theory of Information Retrieval},
  organization={ACM}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/) and the [numpy](https://numpy.org/) package, make sure they are installed.

A file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt datasets_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed.

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```
The following command runs LogOpt where position bias is known, the *--neural* flag indicates a neural model should be used as the logging policy, *-give_prop* indicates that the position bias is known a priori, *--ranker_pair 1* indicates that the first two rankers from the *output/Webscope_C14_Set1/1000rankers.txt* file should be compared:
```
python3 LogOpt.py output/Webscope_C14_Set1/1000rankers.txt local_output/LogOptBiasKnown.txt --ranker_pair 1 --neural --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1 --give_prop
```
The results are stored in *local_output/LogOptBiasKnown.txt*, to replicate the results in Figure 2 and 3 this command has to be repeated 500 times per dataset, where *--ranker_pair* iterates over the numbers 1 to 500.
The same repetition is required for the all of the following commands.

To run LogOpt where position bias is unknown, we remove the *--give_prop* flag, meaning the position bias will be estimated during the gathering of clicks:
```
python3 LogOpt.py output/Webscope_C14_Set1/1000rankers.txt local_output/LogOptBiasUnknown.txt --ranker_pair 1 --neural --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
To simulate an A/B test:
```
python3 ABtest.py output/Webscope_C14_Set1/1000rankers.txt local_output/ABtest.txt --ranker_pair 1 --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
Team Draft Interleaving:
```
python3 TDI.py output/Webscope_C14_Set1/1000rankers.txt local_output/TDI.txt --ranker_pair 1 --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
Probabilistic Interleaving:
```
python3 PI.py output/Webscope_C14_Set1/1000rankers.txt local_output/PI.txt --ranker_pair 1 --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
Optimized Interleaving:
```
python3 OI.py output/Webscope_C14_Set1/1000rankers.txt local_output/OI.txt --ranker_pair 1 --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
Online/Counterfactual evaluation with an Oracle policy (LogOpt with the relevance probabilities known):
```
python3 oraclepolicy.py output/Webscope_C14_Set1/1000rankers.txt local_output/oraclepolicy.txt --ranker_pair 1 --neural --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
To get a uniform logging policy we run the same code with *--update_steps 0* and without *--neural*, this works because it prevents LogOpt from changing the initial uniform random logging policy:
```
python3 oraclepolicy.py output/Webscope_C14_Set1/1000rankers.txt local_output/uniformpolicy.txt --ranker_pair 1 --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1 --update_steps 0
```
To run with a 50/50 A/B logging policy:
```
python3 ABcounterfactual.py output/Webscope_C14_Set1/1000rankers.txt local_output/temp.txt --ranker_pair 1 --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1
```
This is enough to replicate our results in the paper on the same set of rankers.
If you want to generate your own set of rankers you can use the following:
```
python3 rankergeneration.py local_output/temp_rankers.txt --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1 --num_rankers 10
```
This trains 10 rankers on 100 randomly sampled queries using only a random selection of 50% of dataset features.
To see the CTR of each ranker you can use:
```
python3 CTRdistribution.py local_output/temp_rankers.txt local_output/temp_CTR.txt
```
The CTR results are now stored in *local_output/temp_CTR.txt*.
