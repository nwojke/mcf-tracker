# vim: expandtab:ts=4:sw=4
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import generate_detections
import mcf

_ALMOST_INFTY_NUM_TRAJECTORIES = 5000


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _iou(boxes1, boxes2):
    """Computer intersection over union.

    Parameters
    ----------
    boxes1 : ndarray
        The first Nx4 dimensional array of N bounding boxes in
        format (top-left-x, top-left-y, width, height).
    boxes2 : ndarray
        The second Mx4 dimensional array of N bounding boxes in
        format (top-left-x, top-left-y, width, height).

    Returns
    -------
    ndarray
        An NxM dimensional array of pair-wise intersection over union scores
        such that element (i, j) corresponds to the intersection over union
        score between boxes1[i] and boxes2[j]. The score is in [0, 1] and a
        higher score means a larger fraction of boxes[i] is occluded by
        boxes[j].

    """
    intersection_over_union = np.zeros((len(boxes1), len(boxes2)))
    areas1 = boxes1[:, 2:].prod(axis=1)
    areas2 = boxes2[:, 2:].prod(axis=1)

    for i, box in enumerate(boxes1):
        box_tl, box_br = box[:2], box[:2] + box[2:]
        candidates_tl = boxes2[:, :2]
        candidates_br = boxes2[:, :2] + boxes2[:, 2:]

        tl = np.c_[np.maximum(box_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                   np.maximum(box_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(box_br[0], candidates_br[:, 0])[:, np.newaxis],
                   np.minimum(box_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        intersection_over_union[i] = area_intersection / (
            areas1[i] + areas2 - area_intersection)

    return intersection_over_union


def compute_pairwise_transition_features(
        time_gap, boxes1, features1, boxes2, features2):
    """Compute features for matching detections from different time steps.

    This function computes the following 7-dimensional feature vector:

    0-1: Number of frames that have passed between the two detections.
    1-2: Intersection over union bounding box score.
    2-4: Relative size change (for each axis individually)
    4-6: Relative position change (for each axis individually)
    6-7: Appearance descriptor cosine distance

    Parameters
    ----------
    time_gap : int
        It is assumed that all detections in boxes1 have been obtained at the
        same time step. Likewise, all detections in boxes2 have to be obtained
        at the same time step. The time_gap is the number of time steps
        inbetween the two times (successor time index minus predecessor time
        index).
    boxes1 : ndarray
        The first Nx4 dimensional array of N bounding boxes in
        format (top-left-x, top-left-y, width, height).
    features1 : ndarray
        The first NxL dimensional array of N appearance features of
        length L.
    boxes2 : ndarray
        The second Mx4 dimensional array of N bounding boxes in
        format (top-left-x, top-left-y, width, height).
    features2 : ndarray
        The second NxL dimensional array of N appearance features of
        length L.

    Returns
    -------
    ndarray
        The NxMx7 dimensional array of pair-wise transition features such that
        element (i, j) contains the 7-dimensional feature vector for pair
        boxes1[i] and boxes2[j].

    """
    num_objects1 = len(boxes1)
    num_objects2 = len(boxes2)
    assert len(features1) == num_objects1
    assert len(features2) == num_objects2

    features = np.zeros((num_objects1, num_objects2, 7))
    features[:, :, 0:1] = time_gap

    intersection_over_union_score = _iou(boxes1, boxes2)
    features[:, :, 1:2] = intersection_over_union_score[:, :, np.newaxis]

    size1, size2 = boxes1[:, 2:], boxes2[:, 2:]
    max_size = np.maximum(size1[:, np.newaxis], size2[np.newaxis, :])
    features[:, :, 2:4] = np.abs(
        size1[:, np.newaxis] - size2[np.newaxis, :]) / max_size

    positions1 = boxes1[:, :2] + boxes1[:, 2:] / 2.0
    positions2 = boxes2[:, :2] + boxes2[:, 2:] / 2.0
    features[:, :, 4:6] = np.abs(
        positions1[:, np.newaxis] - positions2[np.newaxis, :]) / max_size

    appearance_similarity = _cosine_distance(features1, features2)
    features[:, :, 6:7] = appearance_similarity[:, :, np.newaxis]

    return features


class ObservationCostModel(object):
    """
    The observation cost model computes the cost of adding a detection to any
    object trajectory based on its detector confidence score. The computed cost
    becomes negative if the detection is more likely generated by an object
    than clutter and positive otherwise.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self):
        self._classifier = LogisticRegression()

    def train(self, true_positive_scores, false_alarm_scores):
        """Train observation cost model.

        Parameters
        ----------
        true_positive_scores : array_like
            An array of N detector confidence scores corresponding to true
            object detections.
        false_alarm_scores: array_like
            An array of M detector confidence scores corresponding to false
            alarms.

        """
        train_x = np.r_[true_positive_scores, false_alarm_scores]
        train_y = np.r_[[1] * len(true_positive_scores) +
                        [0] * len(false_alarm_scores)]
        self._classifier.fit(train_x[:, np.newaxis], train_y)

    def compute_cost(self, detector_confidence_scores):
        """Compute observation cost from given detector confidence scores.

        Parameters
        ----------
        detector_confidence_scores : array_like
            An array of N detector confidence scores.

        Returns
        -------
        ndarray
            Returns an array of N costs associated with the given input
            detector confidence scores. The cost becomes negative if a detection
            is more likely generated by an object than clutter.

        """
        # log [p_false_alarm / (1 - p_false_alarm)]
        log_probabilities = self._classifier.predict_log_proba(
            np.asarray(detector_confidence_scores).reshape(-1, 1))
        return log_probabilities[:, 0] - log_probabilities[:, 1]


class TransitionCostModel(object):
    """
    The transition cost model computes the cost of linking two detections on a
    single-object trajectory.

    Parameters
    ----------
    n_estimators : int
        The number of gradient boosting stages to perform. A larger number
        usually results in increased performance at higher computational cost.

    """

    def __init__(self, n_estimators=100):
        self._classifier = GradientBoostingClassifier(
            n_estimators=n_estimators)

    def train(self, positive_pairs, negative_pairs):
        """Train model on pairs of positive and negative detections.

        Parameters
        ----------
        positive_pairs : List[Tuple[int, ndarray, ndarray, ndarray, ndarray]]
            A list of pairs that correspond to neighboring detections on an
            object trajectory. Each list entry contains the following items:

            * Time gap between the two detections (successor time index minus
              predecessor time index)
            * Bounding box coordinates of the predecessor detection in format
              (top-left-x, top-left-y, width, height).
            * Appearance descriptor of the predecessor detection.
            * Bounding box coordinates of the successor detection in format
              (top-left-x, top-left-y, width, height).
            * Appearance descriptor of the successor detection.
        negative_pairs : List[Tuple[int, ndarray, ndarray, ndarray, ndarray]]
            A list of pairs that correspond to two detections of different
            object identities in the same format as positive_pairs.

        """
        # Compute features.
        train_x, train_y = [], []

        for time_gap, box1, feature1, box2, feature2 in positive_pairs:
            train_x.append(
                compute_pairwise_transition_features(
                    time_gap, box1[np.newaxis, :], feature1[np.newaxis, :],
                    box2[np.newaxis, :], feature2[np.newaxis, :]).ravel())
            train_y.append(1)

        for time_gap, box1, feature1, box2, feature2 in negative_pairs:
            train_x.append(
                compute_pairwise_transition_features(
                    time_gap, box1[np.newaxis, :], feature1[np.newaxis, :],
                    box2[np.newaxis, :], feature2[np.newaxis, :]).ravel())
            train_y.append(0)

        # Shuffle data and train classifier.
        indices = np.random.permutation(len(train_x))
        train_x = np.asarray(train_x)[indices, :]
        train_y = np.asarray(train_y)[indices]
        self._classifier.fit(train_x, train_y)

    def compute_cost(self, time_gap, boxes1, features1, boxes2, features2):
        """Compute transition cost from given features.

        Parameters
        ----------
        time_gap : int
            It is assumed that all detections in boxes1 have been obtained at
            the same time step. Likewise, all detections in boxes2 have to be
            obtained at the same time step. The time_gap is the number of time
            steps inbetween the two times (successor time index minus
            predecessor time index).
        boxes1 : ndarray
            The first Nx4 dimensional array of bounding box coordinates in
            format (top-left-x, top-right-y, width, height).
        features1 : ndarray
            The first NxL dimensional array of N appearance features of
            length L.
        boxes2 : ndarray
            The second Mx4 dimensional array of bounding box coordinates in
            format (top-left-x, top-right-y, width, height).
        features2 : ndarray
            The second MxL dimensional array of M appearance features of
            length L.

        Returns
        -------
        ndarray
            Returns the NxM dimensional matrix of element-wise transition costs
            where element (i, j) contains the transition cost between boxes1[i]
            and boxes2[j].

        """
        features = compute_pairwise_transition_features(
            time_gap, boxes1, features1, boxes2, features2)
        log_probabilities = self._classifier.predict_log_proba(
            features.reshape(len(boxes1) * len(boxes2), features.shape[-1]))
        return -log_probabilities[:, 1].reshape(len(boxes1), len(boxes2))


class MinCostFlowTracker(object):
    """
    A multi-object tracker based on the min-cost flow formulation of [1]_.


    [1] Zhang, Li, & Nevatia, (2008): Global data association for multi-object
    tracking using network flows. In Computer Vision and Pattern Recognition,
    2008. (pp. 1-8). IEEE.

    Parameters
    ----------
    entry_exit_cost : float
        A (positive) cost term for starting and ending a trajectory. This is
        a smoothing parameter that trades off continuation of existing
        trajectories against starting new ones. A lower cost results in
        increased fragmentations/shorter trajectories.
    max_num_misses : int
        The maximum number of consecutive misses on each individual object
        trajectory.
    miss_rate : float
        The detector miss rate in [0, 1].
    cnn_model_filename : Optional[str]
        Path to CNN model filename used to compute appearance descriptors. If
        None given, features must be passed on to graph construction
        via `process`.
    cnn_batch_size : Optional[int]
        Batch size of the CNN feature encoder. Must be set if cnn_model_filename
        is not None.
    optimizer_window_len : Optional[int]
        If None, the tracker operates in offline mode such that trajectories are
        computed over the entire observation sequence by
        calling `compute_trajectories()`.
        If not None, the tracker operates in online mode where a fixed-length
        history of frames is optimized at each time step. Results from outside
        of the optimization window are cached to provide a consistent labeling.
    transition_cost_pruning_threshold : float
        A threshold on the transition cost. Edges in the graph that have a
        larger cost than this value are pruned from the graph structure.
    observation_cost_bias : Optional[float]
        A bias term that is added to all observation costs. A value larger than
        zero results in fewer object trajectories. A value smaller than zero
        results in more object trajectories.

    Attributes
    ----------
    entry_exit_cost : float
        A (positive) cost term for starting and ending a trajectory. This is
        a smoothing parameter that trades off continuation of existing
        trajectories against starting new ones. A lower cost results in
        increased fragmentations/shorter trajectories.
    observation_cost_model : ObservationCostModel
        The ObservationCostModel used by this tracker.
    transition_cost_model : TransitionCostModel
        The TransitionCostModel used by this tracker.
    online_mode : bool
        If True, the tracker operates in online mode where `process` returns
        a set of object trajectories at each time step. If False, trajectories
        must be computed explicitly using `compute_trajectories`.
    next_frame_idx : int
        Index of the next frame to be processed. All detections that processed
        in the next call to `process` are annotated with this index. The index
        of the first frame to be processed is zero and the counter is increased
        by one for each consecutive call.
    trajectories : List[List[Tuple[int, ndarray]]]
        If in online_mode, contains the most current set of trajectories. If
        not in online_mode, an empty list.

        Each entry contains the index of the frame at which the detection
        occured (see next_frame_idx) and the bounding box in
        format (top-left-x, top-left-y, width, height).
    """

    def __init__(
            self, entry_exit_cost, observation_cost_model,
            transition_cost_model, max_num_misses=5, miss_rate=0.1,
            cnn_model_filename=None, cnn_batch_size=32,
            optimizer_window_len=None,
            transition_cost_pruning_threshold=-np.log(0.01),
            observation_cost_bias=0.0):
        self.entry_exit_cost = entry_exit_cost
        self.observation_cost_model = observation_cost_model
        self.transition_cost_model = transition_cost_model
        self._observation_cost_bias = observation_cost_bias

        self._max_num_misses = max_num_misses
        self._transition_cost_pruning_threshold = (
            transition_cost_pruning_threshold)

        # An exponential model proportional to miss_rate^{time_gap - 1}
        # in [1, max_num_misses].
        time_gap_to_probability = np.asarray([1e-15] + [
            np.power(miss_rate, time_gap - 1)
            for time_gap in range(1, 2 + max_num_misses)
        ])
        time_gap_to_probability /= time_gap_to_probability[1:].sum()
        self._time_gap_to_cost = -np.log(time_gap_to_probability)

        self._cnn_encoder = (
            generate_detections.create_box_encoder(
                cnn_model_filename, cnn_batch_size)
            if cnn_model_filename is not None else None)

        if optimizer_window_len is None:
            self._graph = mcf.Graph()
            self.online_mode = False
        else:
            self._graph = mcf.BatchProcessing(window_len=optimizer_window_len)
            self.online_mode = True

        self.trajectories = []  # Only used in online mode.

        self.next_frame_idx = 0
        self._nodes_in_timestep = []

    def process(self, boxes, scores, bgr_image=None, features=None):
        """Process one frame of detections.

        Parameters
        ----------
        boxes : ndarray
            An Nx4 dimensional array of bounding boxes in
            format (top-left-x, top-left-y, width, height).
        scores : ndarray
            An array of N associated detector confidence scores.
        bgr_image : Optional[ndarray]
            Optionally, a BGR color image; can be omitted if features is
            not None.
        features : Optional[ndarray]
            Optionally, an NxL dimensional array of N feature vectors
            corresponding to the given boxes. If None given, bgr_image must not
            be None and the tracker must be given a CNN model for feature
            extraction on construction.

        Returns
        -------
        NoneType | List[List[Tuple[int, ndarray]]]
            Returns None if the tracker operates in offline mode. Otherwise,
            returns the set of object trajectories at the current time step.

        """
        # Compute features if necessary.
        if features is None:
            assert self._cnn_encoder is not None, "No CNN model given"
            assert bgr_image is not None, "No input image given"
            features = self._cnn_encoder(bgr_image, boxes)

        # Add nodes to graph for detections observed at this time step.
        observation_costs = (
            self.observation_cost_model.compute_cost(scores)
            if len(scores) > 0 else np.zeros((0, )))
        node_ids = []
        for i, cost in enumerate(observation_costs):
            node_id = self._graph.add(
                cost + self._observation_cost_bias, attributes={
                    "box": boxes[i],
                    "feature": features[i],
                    "frame_idx": self.next_frame_idx
                })
            self._graph.link(self._graph.ST, node_id, self.entry_exit_cost)
            self._graph.link(node_id, self._graph.ST, self.entry_exit_cost)
            node_ids.append(node_id)

        # Link detections to candidate predecessors.
        predecessor_time_slices = (
            self._nodes_in_timestep[-(1 + self._max_num_misses):])
        for k, predecessor_node_ids in enumerate(predecessor_time_slices):
            if len(predecessor_node_ids) == 0 or len(node_ids) == 0:
                continue
            predecessors = [self._graph[x] for x in predecessor_node_ids]
            predecessor_boxes = np.asarray(
                [node["box"] for node in predecessors])
            predecessor_features = np.asarray(
                [node["feature"] for node in predecessors])

            time_gap = len(predecessor_time_slices) - k
            transition_costs = self.transition_cost_model.compute_cost(
                time_gap, predecessor_boxes, predecessor_features, boxes,
                features) + self._time_gap_to_cost[time_gap]

            for i, costs in enumerate(transition_costs):
                for j, cost in enumerate(costs):
                    if cost > self._transition_cost_pruning_threshold:
                        continue
                    self._graph.link(
                        predecessor_node_ids[i], node_ids[j], cost)
        self._nodes_in_timestep.append(node_ids)

        # Compute trajectories if in online-model.
        if self.online_mode:
            self._graph.finalize_timestep()
            trajectories = self._graph.run_search()
            self.trajectories = [[
                (self._graph[x]["frame_idx"], self._graph[x]["box"])
                for x in trajectory
            ] for trajectory in trajectories]

        self.next_frame_idx += 1
        return self.trajectories if self.online_mode else None

    def compute_trajectories(self):
        """Compute trajectories over the entire observation sequence.

        Returns
        -------
        List[List[Tuple[int, ndarray]]]
            Returns the set of object trajectories. Each entry contains the
            index of the frame at which the detection occured (see
            next_frame_idx) and the bounding box in
            format (top-left-x, top-left-y, width, height).

        """
        if self.online_mode:
            return self.trajectories

        solver = mcf.Solver(self._graph)
        trajectories = solver.run_search(
            min_flow=0, max_flow=_ALMOST_INFTY_NUM_TRAJECTORIES)
        trajectories = [[(self._graph[x]["frame_idx"], self._graph[x]["box"])
                         for x in trajectory] for trajectory in trajectories]
        return trajectories
