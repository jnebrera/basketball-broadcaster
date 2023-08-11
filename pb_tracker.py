from norfair import Tracker
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from run_utils import get_main_ball
from soccer import Player
from inference import Converter
from PIL import Image
import numpy as np

class PBTracker:
    def __init__(self):
        self.player_tracker = Tracker(
            distance_function=mean_euclidean,
            distance_threshold=250,
            initialization_delay=3,
            hit_counter_max=90,
        )
        self.ball_tracker = Tracker(
            distance_function=mean_euclidean,
            distance_threshold=150,
            initialization_delay=20,
            hit_counter_max=2000,
        )
        self.motion_estimator = MotionEstimator()

    def create_mask(self, frame, detections):
        mask = np.ones(frame.shape[:2], dtype=frame.dtype)
        margin = 40
        for detection in detections:
            xmin = detection.points[0][0]
            ymin = detection.points[0][1]
            xmax = detection.points[1][0]
            ymax = detection.points[1][1]
            mask[ymin - margin: ymax + margin, xmin - margin: xmax + margin] = 0
        return mask

    def update_motion_estimator(self, detections, frame):
        mask = self.create_mask(frame=frame, detections=detections)
        coord_transformations = self.motion_estimator.update(frame, mask=mask)
        return coord_transformations

    def run(self, frame, player_detections, ball_detections):
        detections = ball_detections + player_detections
        coord_transformations = self.update_motion_estimator(detections, frame)
        player_track_objects = self.player_tracker.update(
            detections=player_detections, coord_transformations=coord_transformations
        )

        ball_track_objects = self.ball_tracker.update(
            detections=ball_detections, coord_transformations=coord_transformations
        )
        player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

        return player_detections, ball_detections

    def draw_detections(self, frame: Image, player_detections, ball_detections) -> Image:
        ball = get_main_ball(ball_detections)
        players = Player.from_detections(detections=player_detections, teams=[])
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )
        if ball:
            frame = ball.draw(frame)

        return frame
