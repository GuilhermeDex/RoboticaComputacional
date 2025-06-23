from enum import Enum

class Command(Enum):
    FIRE = 0      # 'BUTTON'
    SELECT = 2
    RESET = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7

class Controls:
    def __init__(self):
        self.buttons = [0] * 8
        self.quit = False

    def clear_buttons(self):
        self.buttons = [0] * 8

    def input_commands(self, commands):
        for cmd in commands:
            self.buttons[cmd.value] = 1

    def get_button_array(self):
        return self.buttons
