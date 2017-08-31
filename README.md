# mcf-tracker

This repository contains a simple example on how to use the
[mcf](https://github.com/nwojke/mcf) library to implement a multi-object
tracker based on the min-cost flow formulation of [1].

The tracker uses [pymotutils](https://github.com/nwojke/pymotutils) for
reading the dataset and visualization.

## Dependencies

* mcf (added as a git submodule)
* pymotutils (added as a git submodule)
* TensorFlow >= 1.0
* sklearn
* NumPy
* OpenCV

## Installation

The mcf library that is used for solving the tracking problem is implemented in
C++. The project comes with a Makefile to build the library and its
dependencies:

```
git clone https://github.com/nwojke/mcf-tracker.git
make
```

This will also set up the necessary file structure inside the project root. If
the installation was successful you should have a `mcf.so` inside the project
root. In addition, a link to the `pymotutils` package should have been created
and the file `generate_detections.py` should have been copied over from the
[deep_sort](https://github.com/nwojke/deep_sort) project.

Finally, download the provided CNN checkpoint that comes with the `deep_sort`
tracker from
[here](https://owncloud.uni-koblenz.de/owncloud/s/f9JB0Jr7f3zzqs8?path=%2Fresources%2Fnetworks)
and save it under the project root directory. 

### Remark

If available, mcf.so will be built against Python3. If you get an error when
import mcf you might be using a different Python version.

## Demo on MOT16 dataset

The following section shows how to train and run the tracker on the MOT16
dataset. The following code downloads the data from the
[MOTChallenge](http://www.motchallenge.net) project page and creates a
dataset folder inside the project root. You may skip this step if you have
downloaded the dataset already.

```
wget https://motchallenge.net/data/MOT16.zip
unzip MOT16.zip -d MOT16
```

### Train the tracker

The tracker must be trained on the MOT16 training sequences before the
tracking application can be run. If you have followed the instructions above
the training can be started using the following command:

```
python motchallenge_trainer.py
``` 

Note that this can take a while to run. On completion, you should have a
`motchallenge_observation_cost_model.pkl` and
`motchallenge_transition_cost_model.pkl` inside the project root directory.

### Run the tracker

You can now run the tracker on one of the test sequences with the following
command (see `--help` for more options):

```
python motchallenge_tracking_app --mot_dir=./MOT16/test --sequence=MOT16-06 \
    --optimizer_window_len=30
```

The above command runs the tracker in online mode: At each time step, a
fixed-length history of frames (here 30) is optimized. Results from previous
time steps are cached to obtain consistent trajectories.
 
If you want to optimize the entire sequence in one batch run the tracker with
argument `--optimizer_window_len=None`. In this case, you will only see
detections but no tracking output during the first pass through the sequence.

On completion, the results will be written to `output_trajectories.txt` in
MOTChallenge evaluation format and a video of the tracking output will be
stored in `output_video.avi`. 

## Demo on KITTI dataset

There is a Python application `kitti_tracking_app.py` that runs the tracker
on the [KITTI tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)
dataset. It is functional, but poorly document. Feel free to explore it.

## References

[1] Zhang, L., Li, Y., & Nevatia, R. (2008). Global data association for
multi-object tracking using network flows. In IEEE Conference on Computer Vision
and Pattern Recognition (pp. 1-8).
