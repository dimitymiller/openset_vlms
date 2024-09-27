# Testing VLM Classifiers

Here we briefly outline how to use this repository to reproduce the results in the paper.

**Note: This is the first push of code related to the paper and will allow you to reproduce the results in the paper. An updated version of the code that is easier to interface your own methods with will be available by November.**

## Running experiments and saving raw results
See **test_clip.ipynb**, **test_align.ipynb** and **test_imagebind.ipynb**

Use these files to run the experiments for each of the VLM classifiers and save the raw results. These can then be used by other files in the *scripts* folder to generate results and plots. These files will test the VLM classifier with different uncertainty baselines (i.e. softmax, cosine and entropy) and different negative embedding setups, across ImageNet, Food101, Places365 and the GTSRB dataset. Raw results will be saved in the *pred_files* folder inside subfolders for the VLM classifier and dataset being tested.

**Note**: You will need to follow the CLIP, ALIGN and ImageBind instructions for installing these VLM classifiers in order to run the scripts. You will also need to ensure the relevant datasets are installed into the *data* folder.

## Evaluate to find the open-set and closed-set performance metrics (Table 1, Table 3, Table S1, Figure S5, Figure S7)
See **scripts/evaluate.py**

Use this file to print the closed-set and open-set performance metrics for a given experiment -- PR curves, AuPR, AuROC, Precision@95%Recall, Recall@95%Precision, TP count, OSE count and Closed-set accuracy.

You can pass in the following arguments:
- *--file* : the path to the cosine predictions numpy file for an experiment
- *--plots* : adding this will plot the precision-recall curve and uncertainty histograms and save the figures in *figures* folder
- *--auroc* : adding this will print the ROC curve metric 

For example, to reproduce the CLIP results shown in Table 1:

```python scripts/evaluate.py --file pred_files/clip/imagenet/standard_cosine_clip.npy```

## Evaluate the random embedding effectiveness (Figure 2, Figure S3, Figure S4)
See **scripts/random_embedding_effectiveness.ipynb**

This file relies on the results files in the *pred_files* folder.

## Evaluate the impact of set size on performance (Figure 6, Figure S1)
See **scripts/setsize_ablation.ipynb**

This file relies on the results files in the *pred_files* folder.

## Evaluate the closed-set vs open-set performance (Figure 5a, Figure S2)
See **scripts/closedset_vs_openset.ipynb**

This file relies on the results files in the *pred_files* folder.
