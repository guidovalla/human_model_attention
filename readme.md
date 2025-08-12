# Structure of this repo
The CNN fine-tunings are based on **[MMAction2](https://github.com/open-mmlab/mmaction2) repository**, using [its documentation](https://mmaction2.readthedocs.io/en/latest/get_started/overview.html).
After having created a proper environmen the fine-tunings have been launched with:
```console
python tools/train.py ${CONFIG} > output_logs.txt
```
where CONFIG is the path to the MMAction model configuration file, the CNN model config could be found in this (`CNN_finetunings`).
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

Using the customized version of the GradCAM class (`gradcam_utils.py`) and the relative python script for the visualisation (`vis_cam.py`),  it is possible produce a txt file with all the values of the heatmap for a further analysis with:
```console
python tools/visualizations/vis_cam.py ${CONFIG} ${CHECKPOINT} ${VIDEO} --file-url ${PATH}
```
The file produced contains all the values dividing the different frame with the line "Frame N - Heatmap values:". I suggest to produce a .sh file to automatize the video Grad-CAMs collection. For the comparison with human gaze data we converted the Grad-CAM heatmaps to saliency maps summing all the frame values for each video and normalized for the number of frames


# Saliency_comparison
A notebook that aggregates the code for the saliency comparison can be found in (`saliency_metrics_comparison`)








