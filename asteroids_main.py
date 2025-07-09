# asteroids_main.py
import retro
import cv2
import time
import os
import argparse

from asteroids_controls import Controls
from asteroids_bot import AsteroidsBot


def parse_args():
    parser = argparse.ArgumentParser(description="Asteroids Bot Controller")
    parser.add_argument(
        "--rom", type=str, default="Asteroids-Atari2600",
        help="ROM name registered in Gym-Retro"
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Game FPS"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = retro.make(game=args.rom)
    print("Botões disponíveis no ambiente:", env.buttons)
    


    obs, _ = env.reset()
    controls = Controls()
    bot = AsteroidsBot(controls)

    while True:
        start_time = time.perf_counter()
        key = cv2.waitKey(1)

        frame = env.get_screen()
        if frame is None or frame.size == 0:
            print("Empty frame received!")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converter se necessario

        # Processar frame com o bot
        bot.refresh(frame)
        
        env.step(controls.get_button_array())


        if controls.quit:
            break

        elapsed = time.perf_counter() - start_time
        time.sleep(max(0, 1/args.fps - elapsed))

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()