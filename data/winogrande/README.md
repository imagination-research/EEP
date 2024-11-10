# WinoGrande 

Version 1.1 (Sep 16th, 2020)

- - - 

## Data

    ./data/
    ├── train_[xs,s,m,l,xl].jsonl          # training set with differnt sizes
    ├── train_[xs,s,m,l,xl]-labels.lst     # answer labels for training sets
    ├── train_debiased.jsonl               # debiased training set
    ├── train_debiased-labels.lst          # answer labels for debiased training set
    ├── dev.jsonl                          # development set
    ├── dev-labels.lst                     # answer labels for development set
    ├── test.jsonl                         # test set
    ├── sample-submissions-labels.lst      # example submission file for leaderboard    
    └── eval.py                            # evaluation script
    
You can use `train_*.jsonl` for training models and `dev` for validation.
Please note that labels are not included in `test.jsonl`. To evaluate your models on `test` set, make a submission to our [leaderboard](https://winogrande.allenai.org).


## Evaluation

You can use `eval.py` for evaluation on the dev split, which yields `metrics.json`. 

    e.g., python eval.py --preds_file ./YOUR_PREDICTIONS.lst --labels_file ./dev-labels.lst

In the prediction file, each line consists of the predictions (1 or 2) by 5 training sets (ordered by `xs`, `s`, `m`, `l`, `xl`, separated by comma) for each evauation set question. 

     2,1,1,1,1
     1,1,2,2,2
     1,1,1,1,1
     .........
     .........

Namely, the first column is the predictions by a model trained/finetuned on `train_xs.jsonl`, followed by a model prediction by `train_s.jsonl`, ... , and the last (fifth) column is the predictions by a model from `train_xl.jsonl`.
Please checkout a sample submission file (`sample-submission-labels.lst`) for reference.

## Submission to Leaderboard

You can submit your predictions on `test` set to the [leaderboard](http://winogrande.allenai.org).
The submission file must be named as `predictions.lst`. The format is the same as above.  

    
## Reference
If you use this dataset, please cite the following paper:

	@article{sakaguchi2019winogrande,
	    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
	    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
	    journal={arXiv preprint arXiv:1907.10641},
	    year={2019}
	}


## License 

Winogrande dataset is licensed under CC BY 2.0.


## Questions?

You may ask us questions at our [google group](https://groups.google.com/a/allenai.org/forum/#!forum/winogrande).


## Contact 

Email: keisukes[at]allenai.org
