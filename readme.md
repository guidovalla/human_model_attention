# Structure of this repo

*UPDATE:* the `new_experiment` folder contains the preliminary experiments to introduce the gaze info into the loss of the model to more closely align the model's visual attention patterns with those of human participants.

### Kinetics-400

arch     | depth | pretrain | frame length x sample rate | top 1 | top 5 | Flops (G) x views | Params (M) | Model
-------- | ----- | -------- | -------------------------- | ----- | ----- | ----------------- | ---------- | --------------------------------------------------------------------------------------------------
I3D      | R50   | Kinetics-400      | 8x8                        | 73.27 | 90.70 | 37.53 x 3 x 10    | 28.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D\_8x8\_R50.pyth)
X3D      | L     | Kinetics-400     | 16x5                       | 77.44 | 93.31 | 26.64 x 3 x 10    | 6.15       | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D\_L.pyth)
SlowFast | R101  | Kinetics-400      | 16x8                       | 78.70 | 93.61 | 215.61 x 3 x 10   | 53.77      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST\_16x8\_R101_50_50.pyth)
_______________________________________________________________________________________________________________________________________________

The CNN fine-tunings are based on **[MMAction2](https://github.com/open-mmlab/mmaction2) repository**, using [its documentation](https://mmaction2.readthedocs.io/en/latest/get_started/overview.html).
After having created a proper environmen the fine-tunings have been launched with:
```console
python tools/train.py ${CONFIG} > output_logs.txt
```
where CONFIG is the path to the MMAction model configuration file, the CNN model config could be found in this repository in the folder `CNN_finetunings`.
The tests have been done with:
```console
python tools/test.py ${CONFIG} ${CHECKPOINT} --dump _outuput.pkl > output_test_logs.txt
```

## Grad-CAMs
The standard procedure to visualize with MMAction the Grad-CAM heatmap of a video (whose path is specified in VIDEO):
```console
 python tools/visualizations/vis_cam.py ${CONFIG} ${CHECKPOINT} ${VIDEO} --out-filename grad_cam_vis.mp4
```

### Extraction heatmap values

Using the customized version of the GradCAM class `gradcam_utils.py` and the relative python script for the visualisation `vis_cam.py` (both in the folder `CNN_finetunings`),  it is possible produce a txt file with all the values of the heatmap for a further analysis with:
```console
python tools/visualizations/vis_cam.py ${CONFIG} ${CHECKPOINT} ${VIDEO} --file-url ${PATH}
```
The file produced contains all the values dividing the different frames with the line _``` Frame N - Heatmap values: ```_ differentiating them. I suggest to produce a .sh file to automatize the video Grad-CAMs collection. For the comparison with human gaze data we converted the Grad-CAM heatmaps to saliency maps summing all the frame values for each video and normalized for the number of frames


# Saliency_comparison
A notebook that aggregates the code for the saliency comparison can be found in `saliency_metrics_comparison`








