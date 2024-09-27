# Testing VLM Object Detectors

Here we briefly outline how to use this repository to reproduce the object detection results in the paper, as well as how to test new open vocabulary object detectors.

**Note: This is the first push of code related to the paper and will allow you to reproduce the results in the paper. An updated version of the code that is easier to interface your own methods with will be available by November.**

## Testing Protocol
For testing on COCO, we provide ```test_protocol/coco_image_labels.json``` to detail the class text labels to test each image with. It has the following format:
```
{
  <im_name>: {
                'all': ["person", "bicycle", "car", ....],
                'open-set': ["person", "car", ....]
             },
  .........
                  
}

```

- **<im_name>** is a string representing a COCO image for testing. It does not contain any path information, e.g., "000000000139.jpg".
    - **all** is a list of text prompts to simulate testing the image with the entire set of class labels (all COCO class labels).
    - **open-set** is a list of text prompts to simulate when testing with only open-set class labels (only text prompts not present in the image).

You can view how this file was created in ```test_protocol/create_coco_protocol.ipynb```.

## Evaluation Script (Table 2, Table 4, Table S2, Figure S6, Figure S9)
You can calculate the results using the evaluation script.

``` python scripts/evaluate.py --file <path_to_predictions>```

For example, to see the results for OVR-RCNN, you can run:

``` python scripts/evaluate.py --file pred_files/ovrcnn/ovrcnn_standard_coco.json```

Predictions must be saved in the format described below. You can use a number of optional other arguments, including:
- ```--plots``` : will save and visualise the Precision-Recall plots and uncertainty histograms used to calculate the results
- ```--auroc``` : will also print the AuROC metric results
- ```--save``` : will save the results for easy visualisation with the *create_plots.ipynb* script (more below)
- ```--save_dir``` : specifies the directory and name of the json file to save the results in. We include our results file in the ```scripts``` folder

**Note:** you will need to store the COCO dataset, in particular the COCO val2017 dataset in the *data* folder.

## Predictions Format
To use the evaluation script, you should save the predictions from a detector as a json file, containing a python dictionary in the following format:
```
{
    'unc-types': [('scores', 'low'), ....],
    'detections': {
                      <im_name>: {
                                    'all': {
                                              'boxes': [[xmin, ymin, xmax, ymax], ....],
                                              'labels': ["person", ....],
                                              'scores': [0.9995, ....],
                                               ......
                                           },
                                    'open-set': {
                                                  'boxes': [[xmin, ymin, xmax, ymax], ....],
                                                  'labels': ["person", ....],
                                                  'scores': [0.9995, ....],
                                                   ......
                                                },
                                 },
                      .........
                  }
}

```

- **unc-types** is a list of the uncertainty types that should be evaluated. Each uncertainty type is represented by a name and the direction that represents a high uncertainty.
  - each uncertainty type should be present in the 'detections/all' and 'detections/open-set' dictionaries.
  -  e.g. for softmax 'scores', a 'low' score represents high uncertainty.
- **detections** contains the detections from each image when testing with the two evaluation types
  - **<im_name>** is a string representing the name of the image tested. It should not contain any path information, e.g., for a COCO image "000000000139.jpg".
    - **all** is the detections produced when testing with the entire set of class labels
      - **boxes** is a list of the predicted bounding boxes. These must be in the original image coordinate frame.
      - **labels** is a list of the predicted class text labels. These must match what is provided in the dataset test files.
      - **scores** is a list of the predicted softmax scores.
      - additional fields can be entered for other uncertainty types. These should be a list containing a single uncertainty per detection.
    - **open-set** is the detections produced when testing with the open-set class labels. These can be found in the dataset test files.
      - This dictionary should contain the same fields as the **all** dictionary.
     
      
## Visualise and create plots for the results (Figure 4, Figure 5b)
See **scripts/create_plots.ipynb** 

Setup to work with our saved results found in *scripts/detector_results_processed.json*
