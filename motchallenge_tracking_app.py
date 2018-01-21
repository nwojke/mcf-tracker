# vim: expandtab:ts=4:sw=4
import argparse
import pickle
import os

import pymotutils
from pymotutils.contrib.datasets import motchallenge

import min_cost_flow_tracker
import min_cost_flow_pymot


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Min-cost flow tracking")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge train/test directory",
        default="MOT16/test")
    parser.add_argument(
        "--detection_dir",
        help="Path to a directory containingcustom detections. Expected "
        "file name: [sequence_name].txt.", required=False)
    parser.add_argument(
        "--sequence", help="Name of the sequence (e.g. MOT16-01)",
        required=True)
    parser.add_argument(
        "--cnn_model",
        default="mars-small128.ckpt-68577",
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
        default=None)
    parser.add_argument(
        "--entry_exit_cost",
        help="A cost term for starting and ending trajectories. A lower cost "
        "results in increased fragmentations/shorter trajectories.",
        type=float, default=10.0)
    parser.add_argument(
        "--observation_cost_bias",
        help="A bias term that is added to all observation costs. A value "
        "larger than zero results in more object trajectories. A value smaller "
        "than zero results in fewer object trajectories.",
        type=float, default=0.0)
    parser.add_argument(
        "--max_num_misses",
        help="Maximum number of consecutive misses before a track should be "
             "dropped.",
        type=int, default=5)
    parser.add_argument(
        "--miss_rate", help="Detector miss rate in [0, 1]", type=float,
        default=0.3)
    parser.add_argument(
        "--optimizer_window_len",
        help="If not None, the tracker operates in online mode where a "
        "fixed-length history of frames is optimized at each time step. If "
        "None, trajectories are computed over the entire sequence (offline "
        "mode).", type=int, default=30)
    parser.add_argument(
        "--show_output", help="If True, shows tracking output", type=bool,
        default=True)
    parser.add_argument(
        "--output_dir", help="Path to tracking output directory. Results "
        "are stored in this directory in MOTChallenge format.",
        default=".")
    return parser.parse_args()


def draw_online_tracking_results(image_viewer, frame_data, pymot_adapter):
    del frame_data  # Unused variable.
    image_viewer.thickness = 2

    newest_frame_idx = pymot_adapter.compute_next_frame_idx() - 1
    for i, trajectory in pymot_adapter.trajectory_dict.items():
        if len(trajectory) == 0 or trajectory[-1].frame_idx != newest_frame_idx:
            continue  # This trajectory is empty or has been terminated.
        image_viewer.color = pymotutils.create_unique_color_uchar(i)
        x, y, w, h = trajectory[-1].sensor_data
        image_viewer.rectangle(x, y, w, h, label="%d" % i)


def main():
    """ Program main entry point.
    """
    args = parse_args()
    devkit = motchallenge.Devkit(args.mot_dir, args.detection_dir)
    data_source = devkit.create_data_source(args.sequence)
    data_source.apply_nonmaxima_suppression(max_bbox_overlap=0.5)

    with open(args.observation_cost_model, "rb") as f:
        observation_cost_model = pickle.load(f)
    with open(args.transition_cost_model, "rb") as f:
        transition_cost_model = pickle.load(f)

    tracker = min_cost_flow_tracker.MinCostFlowTracker(
        args.entry_exit_cost, observation_cost_model, transition_cost_model,
        args.max_num_misses, args.miss_rate, args.cnn_model,
        optimizer_window_len=args.optimizer_window_len,
        observation_cost_bias=args.observation_cost_bias)
    pymot_adapter = min_cost_flow_pymot.PymotAdapter(tracker)

    # Compute a suitable window shape.
    image_shape = data_source.peek_image_shape()[::-1]
    aspect_ratio = float(image_shape[0]) / image_shape[1]
    window_shape = int(aspect_ratio * 600), 600

    visualization = pymotutils.MonoVisualization(
        update_ms=25, window_shape=window_shape,
        online_tracking_visualization=draw_online_tracking_results)

    application = pymotutils.Application(data_source)
    application.process_data(pymot_adapter, visualization)
    application.compute_trajectories(interpolation=True)
    if args.show_output:
        visualization.enable_videowriter(
            os.path.join(args.output_dir, "%s.avi" % args.sequence))
        application.play_hypotheses(visualization)

    if args.output_dir is not None:
        pymotutils.motchallenge_io.write_hypotheses(
            os.path.join(args.output_dir, "%s.txt" % args.sequence),
            application.hypotheses)


if __name__ == "__main__":
    main()
