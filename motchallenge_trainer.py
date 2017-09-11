# vim: expandtab:ts=4:sw=4
import os
import argparse
import pickle
import matplotlib.pyplot as plt

import pymotutils
from pymotutils.contrib.datasets import motchallenge

import generate_detections
import min_cost_flow_pymot


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Min-cost flow tracking")
    parser.add_argument(
        "--mot_train_dir", help="Path to MOTChallenge train directory",
        default="./MOT16/train")
    parser.add_argument(
        "--validation_sequences",
        help="Comma separated list of sequences used for validation",
        default="")
    parser.add_argument(
        "--detection_dir",
        help="Path to a directory containingcustom detections. Expected "
        "file name: [sequence_name].txt.", required=False)
    parser.add_argument(
        "--min_confidence", help="Detector confidence threshold", type=float,
        default=None)
    parser.add_argument(
        "--max_num_misses",
        help="Maximum number of consecutive misses before a track should be "
        "dropped.", type=int, default=5)
    parser.add_argument(
        "--n_estimators", help="Number of gradient boosting stages", type=int,
        default=200)
    parser.add_argument(
        "--cnn_model", default="mars-small128.ckpt-68577",
        help="Path to CNN checkpoint file")
    return parser.parse_args()


def main():
    """ Program main entry point.
    """
    args = parse_args()
    test_sequences = [
        x.strip() for x in args.validation_sequences.split(",")
        if len(x) > 0 and os.path.isdir(os.path.join(args.mot_train_dir, x))]
    train_sequences = [
        x for x in os.listdir(args.mot_train_dir)
        if os.path.isdir(os.path.join(args.mot_train_dir, x)) and
        x not in test_sequences
    ]

    print("Training sequences: %s" % ", ".join(train_sequences))
    print("Validation sequences: %s" % ", ".join(test_sequences))

    feature_extractor = generate_detections.create_box_encoder(args.cnn_model)
    devkit = motchallenge.Devkit(args.mot_train_dir, args.detection_dir)
    trainer = min_cost_flow_pymot.MinCostFlowTrainer()

    for i, sequence in enumerate(train_sequences):
        print("Processing %s" % sequence)
        data_source = devkit.create_data_source(sequence, args.min_confidence)

        print("\tComputing features ...")
        pymotutils.compute_features(
            data_source.detections, data_source.bgr_filenames,
            feature_extractor)

        print("\tCollecting training data ...")
        trainer.add_dataset(
            data_source.ground_truth, data_source.detections,
            args.max_num_misses)

    print("Training observation cost model ...")
    observation_cost_model = trainer.train_observation_cost_model()
    with open("motchallenge_observation_cost_model.pkl", "wb") as f:
        pickle.dump(observation_cost_model, f)

    print("Training transition cost model ...")
    transition_cost_model = trainer.train_transition_cost_model(
        args.n_estimators)
    with open("motchallenge_transition_cost_model.pkl", "wb") as f:
        pickle.dump(transition_cost_model, f)

    print("Done")
    if len(test_sequences) == 0:
        return

    print("Creating plots on validation sequences.")

    for sequence in test_sequences:
        print("Processing %s" % sequence)
        data_source = devkit.create_data_source(sequence, args.min_confidence)

        print("\tComputing features ...")
        pymotutils.compute_features(
            data_source.detections, data_source.bgr_filenames,
            feature_extractor)

        print("\tComputing validation scores ...")
        eval_time_gaps = [1, 1 + args.max_num_misses]

        scores = min_cost_flow_pymot.score_dataset(
            data_source.ground_truth, data_source.detections, eval_time_gaps,
            observation_cost_model, transition_cost_model)

        print("\tCalling plotting routines ...")
        figure_handle = plt.figure()
        figure_handle.suptitle("%s (Observation Cost)" % sequence)
        axes = figure_handle.add_subplot(121)
        axes.set_title("Positive Cost")
        axes.hist(scores[0], bins=10)
        axes = figure_handle.add_subplot(122)
        axes.set_title("Negative Cost")
        axes.hist(scores[1], bins=10)

        for time_gap in eval_time_gaps:
            figure_handle = plt.figure()
            plt.suptitle("%s (Time Gap %d)" % (sequence, time_gap))
            axes = figure_handle.add_subplot(121)
            axes.set_title("Positive Cost")
            axes.hist(scores[2][time_gap])
            axes = figure_handle.add_subplot(122)
            axes.set_title("Negative Cost")
            axes.hist(scores[3][time_gap])

    print("Done.")
    if len(test_sequences) > 0:
        plt.show()


if __name__ == "__main__":
    main()
