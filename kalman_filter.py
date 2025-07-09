import numpy as np
import cv2

class KalmanTracker:
    def __init__(self, initial_position):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[initial_position[0]],
                                         [initial_position[1]],
                                         [0],
                                         [0]], np.float32)

    def correct(self, position):
        measurement = np.array([[np.float32(position[0])],
                                [np.float32(position[1])]])
        self.kalman.correct(measurement)

    def predict(self):
        prediction = self.kalman.predict()
        return [int(prediction[0]), int(prediction[1])]
