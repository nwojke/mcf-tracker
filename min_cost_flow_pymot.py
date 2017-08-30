# vim: expandtab:ts=4:sw=4
import numpy as np

import pymotutils
import min_cost_flow_tracker


def collect_positive_negative_pairs(track_set, time_gaps):
    """Collect positive and negative training pairs for transition cost model.

    Parameters
    ----------
    track_set : pymotutils.TrackSet
        A ground-truth track set where each detection contains a bounding box
        attribute `roi` in format (top-left-x, top-left-y, width, height)` and
        an appearance descriptor attribute `feature`.
    time_gaps : Iterable[int]
        A list of time offsets to be used for generating pairs of detections.
        The time offset specifies the number of time steps between the two
        observed detections (must be at least 1). For example, by passing in [1]
        all pairs of detections have been observed in consecutive time steps.

    Returns
    -------
    Tuple[List[Tuple[int, ndarray, ndarray, ndarray, ndarray]], List[Tuple[int, ndarray, ndarray, ndarray, ndarray]]]
        Returns the positive and negative pairs where a positive pair shows two
        detections that belong to the same object and a negative pair shows two
        detections that belong to different objects.

        Each element in the respective list of pairs contains:

        * The time_gap
        * The bounding box and feature of the first detection
        * The bounding box and feature of the second detection

    """
    positive_pairs, negative_pairs = [], []

    for time_gap in time_gaps:

        def iterate_callback(track_id_i, detection_i, track_id_j, detection_j):
            detection_pair = (
                time_gap, detection_i.roi, detection_i.feature,
                detection_j.roi, detection_j.feature)

            if track_id_i == track_id_j:
                positive_pairs.append(detection_pair)
            else:
                negative_pairs.append(detection_pair)

        pymotutils.iterate_track_set_with_time_offset(
            track_set, time_offset=time_gap, for_each=iterate_callback)

    return positive_pairs, negative_pairs


class MinCostFlowTrainer(object):
    """
    A convenience class to train observation and transition cost models.
    """

    def __init__(self):
        self._positive_scores = []
        self._negative_scores = []

        self._positive_pairs = []
        self._negative_pairs = []

    def add_dataset(self, ground_truth, detections, max_num_misses):
        """Add a dataset to the set of training examples.

        Parameters
        ----------
        ground_truth : pymotutils.TrackSet
            The ground-truth track set. Detections should contain the
            bounding box in format (top-left-x, top-left-y, width, height)
            in the `sensor_data` field.
        detections : Dict[int, List[pymotutils.RegionOfInterestDetection]]
            A dictionary that maps from time step to list of detections.
        max_num_misses : int
            The method generates pairs of detections on each trajectory. Theses
            pairs are at most `1 + max_num_misses` time steps apart.

        """
        track_set, false_alarms = pymotutils.associate_detections(
            ground_truth, detections)

        for track in track_set.tracks.values():
            self._positive_scores += [
                d.confidence for d in track.detections.values()
            ]
        for false_alarm_list in false_alarms.values():
            self._negative_scores += [d.confidence for d in false_alarm_list]

        positive_pairs, negative_pairs = collect_positive_negative_pairs(
            track_set, range(1, 1 + max_num_misses))
        self._positive_pairs += positive_pairs
        self._negative_pairs += negative_pairs

    def train_observation_cost_model(self):
        """Train observation cost model from given detections.

        Returns
        -------
        min_cost_flow_tracker.ObservationCostModel
            Returns an observation cost model that has been trained on the
            given training data.

        """
        model = min_cost_flow_tracker.ObservationCostModel()
        model.train(self._positive_scores, self._negative_scores)
        return model

    def train_transition_cost_model(self, n_estimators=100):
        """Train transition cost model from given detections.

        Parameters
        ----------
        n_estimators : int
            Number of gradient boosting stages to perform. A larger number
            usually results in increased performance at higher computational
            cost.

        Returns
        -------
        min_cost_flow_tracker.TransitionCostModel
            Returns a transition cost model that has been trained on the
            given detections.

        """
        model = min_cost_flow_tracker.TransitionCostModel(n_estimators)
        model.train(self._positive_pairs, self._negative_pairs)
        return model


def score_dataset(
        ground_truth, detections, time_gaps, observation_cost_model,
        transition_cost_model):
    """Evaluate cost models on validation dataset.

    Parameters
    ----------
    ground_truth : pymotutils.TrackSet
    detections : Dict[int, List[pymotutils.RegionOfInterestDetection]]
    time_gaps : List[int]
        The transition cost model is evaluated on pairs of detections belonging
        to the same/different objects. The time_gaps define the difference in
        time from which these pairs are genererated. For example, by passing
        in `[1]` all pairs of detections have been observed in neighboring
        time steps.
    observation_cost_model : min_cost_flow_tracker.ObservationCostModel
        The observation cost model to evaluate.
    transition_cost_model : min_cost_flow_tracker.TransitionCostModel
        The transition cost model to evaluate.

    Returns
    -------
    ndarray, ndarray, Dict[int, ndarray], Dict[int, ndarray]
        The first element of the tuple contains the assigned cost of the
        observation cost model for detections that correspond to true
        objects (not clutter). The second element contains the assigned cost
        for false alarms.

        The third and fourth elements of the tuple contain dictonaries that
        map from time_gap (as passed in by the corresponding parameter) to
        the cost assigned by the transition cost model to pairs of detections
        showing the same/different identities.

    """
    track_set, false_alarms = pymotutils.associate_detections(
        ground_truth, detections)

    # Evaluate observation cost model.
    positive_scores, negative_scores = [], []
    for track in track_set.tracks.values():
        positive_scores += [d.confidence for d in track.detections.values()]
    for false_alarm_list in false_alarms.values():
        negative_scores += [d.confidence for d in false_alarm_list]

    positive_observation_costs = (
        observation_cost_model.compute_cost(positive_scores))
    negative_observation_costs = (
        observation_cost_model.compute_cost(negative_scores))

    # Evaluate transition cost model.
    positive_pairs, negative_pairs = collect_positive_negative_pairs(
        track_set, time_gaps)

    positive_transition_costs = {}
    for time_gap in time_gaps:
        pairs = [x for x in positive_pairs if x[0] == time_gap]
        positive_transition_costs[time_gap] = [
            transition_cost_model.compute_cost(
                time_gap, box1[np.newaxis, :], feature1[np.newaxis, :],
                box2[np.newaxis, :], feature2[np.newaxis, :])[0, 0]
            for _, box1, feature1, box2, feature2 in pairs
        ]

    negative_transition_costs = {}
    for time_gap in time_gaps:
        pairs = [x for x in negative_pairs if x[0] == time_gap]
        negative_transition_costs[time_gap] = [
            transition_cost_model.compute_cost(
                time_gap, box1[np.newaxis, :], feature1[np.newaxis, :],
                box2[np.newaxis, :], feature2[np.newaxis, :])[0, 0]
            for _, box1, feature1, box2, feature2 in pairs
        ]

    return (
        positive_observation_costs, negative_observation_costs,
        positive_transition_costs, negative_transition_costs)


class PymotAdapter(pymotutils.Tracker):
    """
    An adapter class that implements the required interface of the
    pymotutils.Tracker class to wrap an existing
    min_cost_flow_tracker.MinCostFlowTracker.

    Parameters
    ----------
    tracker : min_cost_flow_tracker.MinCostFlowTracker
        The tracker to be wrapped.

    """

    def __init__(self, tracker):
        assert isinstance(tracker, min_cost_flow_tracker.MinCostFlowTracker)
        self.tracker = tracker
        self._start_idx = 0

    def reset(self, start_idx, end_idx):
        del end_idx  # Unused variable
        self._start_idx = start_idx

    def process_frame(self, frame_data):
        detections = frame_data["detections"]

        features = [d.feature for d in detections]
        if any(feature is None for feature in features):
            # No pre-computed features, let the tracker handle this situation.
            features = None
        bgr_image = (
            frame_data["bgr_image"] if "bgr_image" in frame_data else None)
        boxes = np.asarray([d.roi for d in detections])
        scores = np.asarray([d.confidence for d in detections])

        self.tracker.process(boxes, scores, bgr_image, features)

    def compute_trajectories(self):
        trajectories = self.tracker.compute_trajectories()
        trajectories = [[
            pymotutils.Detection(
                frame_idx=self._start_idx + x[0], sensor_data=x[1])
            for x in trajectory
        ] for trajectory in trajectories]
        return trajectories
