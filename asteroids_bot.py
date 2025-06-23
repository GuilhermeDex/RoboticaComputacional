import cv2
import numpy as np
import math
import time
from skimage.measure import label, regionprops
from asteroids_elements import Ship, Asteroid
from asteroids_controls import Command


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


class Asteroid:
    color = ([0, 0, 50], [180, 60, 255])  # HSV

    def __init__(self, position):
        self.position = position
        self.tracker = KalmanTracker(position)

    def update(self, new_position):
        self.tracker.correct(new_position)
        self.position = new_position

    def predict(self):
        return self.tracker.predict()


class AsteroidsBot:
    def __init__(self, controls):
        self.controls = controls
        self.ship = None
        self.asteroids = []
        self.ship_angle = 90  # Começa apontando para cima
        self.fire_cooldown = 0

        self.last_ship_time = 0
        self.last_asteroids_time = 0
        self.ship_timeout = 0.3  # segundos
        self.asteroid_timeout = 0.3

    def refresh(self, frame):
        self.detect_elements(frame)
        self.action()

    def detect_elements(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            now = time.time()

            # --- Detecta Nave ---
            lower_ship = np.array([0, 100, 100])
            upper_ship = np.array([10, 150, 255])
            ship_mask = cv2.inRange(hsv, lower_ship, upper_ship)

            contours, _ = cv2.findContours(ship_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                center = [x + w // 2, y + h // 2]
                self.ship = Ship(center)
                self.last_ship_time = now
                cv2.circle(frame, tuple(center), 4, (0, 255, 0), -1)

            # --- Detecta Asteroides ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, asteroid_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            asteroid_mask = cv2.bitwise_and(asteroid_mask, cv2.bitwise_not(ship_mask))

            kernel = np.ones((3, 3), np.uint8)
            asteroid_mask = cv2.morphologyEx(asteroid_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(asteroid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_asteroids = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 6:
                    x, y, w, h = cv2.boundingRect(cnt)
                    center = [x + w // 2, y + h // 2]
                    detected_asteroids.append(Asteroid(center))
                    cv2.drawContours(frame, [cnt], -1, (255, 255, 255), 1)

            if detected_asteroids:
                self.asteroids = detected_asteroids
                self.last_asteroids_time = now

            # --- Timeout ---
            if now - self.last_ship_time > self.ship_timeout:
                self.ship = None
            if now - self.last_asteroids_time > self.asteroid_timeout:
                self.asteroids = []

            # Debug visual
            cv2.imshow("Ship Mask", ship_mask)
            cv2.imshow("Asteroid Mask", asteroid_mask)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Erro na detecção: {str(e)}")
            self.ship = None
            self.asteroids = []

    def action(self):

        if not self.ship or not self.asteroids:
            print("Sem nave ou asteroides detectados")
            self.controls.clear_buttons()
            return

        # evasao
        danger_zone = 25  
        evade_actions = []

        for asteroid in self.asteroids:
            future_pos = asteroid.predict()
            distance = np.linalg.norm(np.array(future_pos) - np.array(self.ship.position))

            if distance < danger_zone:
                dx = self.ship.position[0] - future_pos[0]
                dy = self.ship.position[1] - future_pos[1]
                angle_away = math.degrees(math.atan2(-dy, dx)) % 360

                diff = (angle_away - self.ship_angle + 360) % 360
                if diff > 180:
                    diff -= 360

                ROTATION_STEP = 10
                TOLERANCE = 15

                if diff > TOLERANCE:
                    evade_actions.append(Command.RIGHT)
                    self.ship_angle = (self.ship_angle + ROTATION_STEP) % 360
                elif diff < -TOLERANCE:
                    evade_actions.append(Command.LEFT)
                    self.ship_angle = (self.ship_angle - ROTATION_STEP) % 360
                else:
                    evade_actions.append(Command.UP)

                break

        if evade_actions:
            self.controls.clear_buttons()
            self.controls.input_commands(evade_actions)
            return

        # Ataque ao asteroide
        closest = min(self.asteroids, key=lambda ast: self.ship.distance_to(ast))
        dx = closest.position[0] - self.ship.position[0]
        dy = closest.position[1] - self.ship.position[1]
        angle_to_asteroid = math.degrees(math.atan2(-dy, dx)) % 360

        diff = (angle_to_asteroid - self.ship_angle + 360) % 360
        if diff > 180:
            diff -= 360

        print(f" Angulo nave: {self.ship_angle:.1f}° | Alvo: {angle_to_asteroid:.1f}° | Diferenca: {diff:.1f}°")

        actions = []
        ROTATION_STEP = 9
        TOLERANCE = 20
        
        actions.append(Command.UP)

        if abs(diff) <= TOLERANCE:
            if self.fire_cooldown <= 0:
                actions.append(Command.FIRE)
                self.fire_cooldown = 5
            else:
                self.fire_cooldown -= 1
        else:
            self.fire_cooldown = max(self.fire_cooldown - 1, 0)    
            
        if diff > TOLERANCE:
            actions.append(Command.RIGHT)
            self.ship_angle = (self.ship_angle + ROTATION_STEP) % 360
        elif diff < -TOLERANCE:
            actions.append(Command.LEFT)
            self.ship_angle = (self.ship_angle - ROTATION_STEP) % 360
        else:
            actions.append(Command.FIRE)

        print("Ações:", actions)
        self.controls.clear_buttons()
        self.controls.input_commands(actions)
        print("Botões aplicados:", self.controls.buttons)
