# vim: expandtab:ts=4:sw=4
import argparse
import pickle

import pymotutils
from pymotutils.contrib.datasets import kitti

import min_cost_flow_tracker
import min_cost_flow_pymot


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Min-cost flow tracking")
    parser.add_argument(
        "--kitti_dir", help="Path to KITTI training/testing directory")
    parser.add_argument(
        "--sequence", help="A four digit sequence number")
    parser.add_argument(
        "--cnn_model", default="mars-small128.ckpt-68577",
        help="Path to CNN checkpoint file")
    parser.add_argument(
        "--observation_cost_model",
        help="Path to pickled observation cost model",
        default="motchallenge_observation_cost_model.pkl")
    parser.add_argument(
        "--transition_cost_model",
        help="Path to pickled transition cost model",
        default="motchallenge_transition_cost_model.pkl")
    parser.add_argument(
        "--min_confidence", help="Detector confidence threshold", type=float,
        default=-1.0)
    parser.add_argument(
        "--entry_exit_cost",
        help="A cost term for starting and ending trajectories. A lower cost "
        "results in increased fragmentations/shorter trajectories.",
        type=float, default=10.0)
    parser.add_argument(
        "--max_num_misses",
        help="The maximum number of consecutive misses on each individual "
        "object trajectory.", type=int, default=5)
    parser.add_argument(
        "--miss_rate", help="Detector miss rate in [0, 1]", type=float,
        default=0.3)
    parser.add_argument(
        "--optimizer_window_len",
        help="If not None, the tracker operates in online mode where a "
        "fixed-length history of frames is optimized at each time step. If "
        "None, trajectories are computed over the entire sequence (offline "
        "mode).", type=int, default=30)
    return parser.parse_args()


def draw_online_tracking_results(image_viewer, frame_data, pymot_adapter):
    del frame_data  # Unused variable.
    image_viewer.thickness = 2

    newest_frame_idx = pymot_adapter.tracker.next_frame_idx - 1
    for i, trajectory in enumerate(pymot_adapter.tracker.trajectories):
        if len(trajectory) == 0 or trajectory[-1][0] != newest_frame_idx:
            continue  # This trajectory is empty or has been terminated.
        image_viewer.color = pymotutils.create_unique_color_uchar(i)
        x, y, w, h = trajectory[-1][1]
        image_viewer.rectangle(x, y, w, h, label="%d" % i)


def main():
    """ Program main entry point.
    """
    args = parse_args()
    devkit = kitti.Devkit(args.kitti_dir)
    data_source = devkit.create_data_source(
        args.sequence, kitti.OBJECT_CLASSES_PEDESTRIANS,
        min_confidence=args.min_confidence)

    with open(args.observation_cost_model, "rb") as f:
        observation_cost_model = pickle.load(f)
    with open(args.transition_cost_model, "rb") as f:
        transition_cost_model = pickle.load(f)

    tracker = min_cost_flow_tracker.MinCostFlowTracker(
        args.entry_exit_cost, observation_cost_model, transition_cost_model,
        args.max_num_misses, args.miss_rate, args.cnn_model,
        optimizer_window_len=args.optimizer_window_len)
    pymot_adapter = min_cost_flow_pymot.PymotAdapter(tracker)

    visualization = pymotutils.MonoVisualization(
        update_ms=kitti.CAMERA_UPDATE_IN_MS,
        window_shape=kitti.CAMERA_IMAGE_SHAPE,
        online_tracking_visualization=draw_online_tracking_results)

    application = pymotutils.Application(data_source)
    visualization.enable_videowriter("/tmp/detections.avi")
    application.process_data(pymot_adapter, visualization)
    application.compute_trajectories(interpolation=True)
    visualization.enable_videowriter("/tmp/trajectories.avi")
    application.play_hypotheses(visualization)


if __name__ == "__main__":
    main()
