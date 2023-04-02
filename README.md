# CNN-based human Detection using Infrared Images for Night surveillance


## Tech Stack

- TensorFlow
- Scikit
- Pillowt
- Matlabplot.lib
- h5py

## Getting Started

To get started:

- Clone the repo.

```shell
git clone https://github.com/sanz17/Human-detection-at-night-using-CNN.git
```

- Change into the directory.

```shell
cd Human-detection-at-night-using-CNN
```
- Run using VsCode.

```shell
code .
```

```
python annotate.py --dataset university --margin 50
```

## Baselines

Run the baseline algorithms and compute the resulting accuracy.
"--method" can be one of "threshold", "threshold\_adp", "backSub" or "kmeans"
Result images will be displayed if the "--viz" flag is used, otherwise only the accuracy is computed.
Use the "--save_frame" flag to save results from a specific frame to the "results" folder.

```
python baselines.py --dataset university --method threshold --viz --save_frame 100 --min_cluster 200
```

## Convolutional Neural Network

First, the picture files and label files are converted into H5 files for use in training. If the "—use history" option is present, input channels will include three components: I the intensity of the infrared picture (ii) the difference image for one time step (iii) the output image after executing background subtraction Before proceeding, run "baselines.py" with "—method backSub" to save results of background subtraction in the "backSub" folder. If not, just the first component is used.

```
python process_record.py --dataset university --imsize 385
```

Train the network using "train.py" (150 epochs). The trained model will be saved in "dataset/myfolder/model.ckpt".

```
python train.py --dataset university
```

Test the network and measure the accuracy. The detection threshold (a number between 0 and 1) controls 
the confidence level above which a pixel will be considered a positive detection.
Result images will be displayed if the "--viz" flag is used, otherwise only the accuracy is computed.

```
python test.py --dataset university --imsize 385 --detection_threshold 0.99 --viz --min_cluster 200 --save
```

Evaluate the accuracy of a trained model. The input folder should contain the model checkpoint, label images and prediction images.

```
python evaluate.py dataset/university
```
