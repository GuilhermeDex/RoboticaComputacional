import cv2
import numpy as np
import math
import time
from asteroids_elements import Ship
from asteroids_controls import Command
from kalman_filter import KalmanTracker
from qlearning_agent import QLearningAgent


class Asteroid:
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

        # Q-Learning
        self.actions_list = [
            [Command.LEFT],
            [Command.RIGHT],
            [Command.UP],
            [Command.FIRE],
            [Command.UP, Command.FIRE],
            [Command.LEFT, Command.FIRE],
            [Command.RIGHT, Command.FIRE],
            [Command.LEFT, Command.UP],
        ]
        self.agent = QLearningAgent(actions=self.actions_list)
        self.prev_state = None
        self.prev_action = None

    def refresh(self, frame):
        ship_alive_before = self.ship is not None
        self.detect_elements(frame)
        ship_alive_after = self.ship is not None

        # Verifica se a nave "morreu"
        if ship_alive_before and not ship_alive_after:
            print("[⚠️] Nave destruída!")
            self.penalize_death()

        self.action()

    def detect_elements(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            now = time.time()

            # Detecta Nave
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

            # Detecta Asteroides
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, asteroid_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            asteroid_mask = cv2.bitwise_and(asteroid_mask, cv2.bitwise_not(ship_mask))

            kernel = np.ones((3, 3), np.uint8)
            asteroid_mask = cv2.morphologyEx(asteroid_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(asteroid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_asteroids = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 3:
                    x, y, w, h = cv2.boundingRect(cnt)
                    center = [x + w // 2, y + h // 2]
                    detected_asteroids.append(Asteroid(center))
                    cv2.drawContours(frame, [cnt], -1, (255, 255, 255), 1)

            if detected_asteroids:
                self.asteroids = detected_asteroids
                self.last_asteroids_time = now

            if now - self.last_ship_time > self.ship_timeout:
                self.ship = None
            if now - self.last_asteroids_time > self.asteroid_timeout:
                self.asteroids = []

            for asteroid in self.asteroids:
                predicted = asteroid.predict()
                cv2.circle(frame, tuple(predicted), 2, (0, 0, 255), -1)

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

        closest = min(self.asteroids, key=lambda ast: self.ship.distance_to(ast))
        dx = closest.position[0] - self.ship.position[0]
        dy = closest.position[1] - self.ship.position[1]
        angle_to_asteroid = math.degrees(math.atan2(-dy, dx)) % 360

        diff = (angle_to_asteroid - self.ship_angle + 360) % 360
        if diff > 180:
            diff -= 360

        distance = np.linalg.norm(np.array(closest.position) - np.array(self.ship.position))

        # Estado atual
        state = (
            int(self.ship_angle / 30),
            int(angle_to_asteroid / 30),
            int(distance / 50)
        )

        instintivo = False
        # Comportamento instintivo de sobrevivência
        if distance < 30 and abs(diff) < 20:
            print("Instinto ativado: FUGA")
            actions = [Command.LEFT, Command.UP]
            instintivo = True
        else:
            action_idx = self.agent.choose_action(state)
            actions = self.actions_list[action_idx]

        # Recompensa
        reward = 0
        if abs(diff) <= 10:
            reward += 1
        if Command.FIRE in actions:
            reward += 0.5  # recompensado por atirar
        if distance < 25 and Command.FIRE in actions:
            reward += 2  # potencial de acertar
        if self.prev_action == self.actions_list.index(actions):
            reward -= 0.5  # punição por inatividade
        if instintivo:
            reward -= 1  

        if not instintivo:
            if self.prev_state is not None and self.prev_action is not None:
                self.agent.update(self.prev_state, self.prev_action, reward, state)
            self.prev_state = state
            self.prev_action = self.actions_list.index(actions)

        # print(f" Angulo nave: {self.ship_angle:.1f}° | Alvo: {angle_to_asteroid:.1f}° | Diferenca: {diff:.1f}°")
        print("Ações:", actions)

        self.controls.clear_buttons()
        self.controls.input_commands(actions)
        print("Botões aplicados:", self.controls.buttons)

    def penalize_death(self):
        if self.prev_state is not None and self.prev_action is not None:
            # Penalidade forte pela morte
            self.agent.update(self.prev_state, self.prev_action, reward=-10, next_state=self.prev_state)
