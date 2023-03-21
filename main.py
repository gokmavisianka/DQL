import pygame
import numpy as np
from datetime import datetime
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


pygame.init()

FPS = 5

# +--------------------------------------------------------------------+
# |        [1: up]                                                     |
# | [0: left] O [2: right]                                             |
# |       [3: down]                                                    |
# |                                                                    |
# | Actions: [0: Turn Left] [1: Straight] [2: Turn Right]              |
# |                                                                    |
# | if action is 0, reduce the number that represents the direction.   |
# |                                                                    |
# | Current direction: 1 (up)                                          |
# | current action: 0 (turn left)                                      |
# | New direction: 1 - 1 = 0 (left)                                    |
# |                                                                    |
# | if action is 2, increase the number that represents the direction. |
# |                                                                    |
# | Current direction: 1 (up)                                          |
# | current action: 2 (turn right)                                     |
# | New direction: 1 + 1 = 2 (right)                                   |
# +--------------------------------------------------------------------+

# If direction is 0 and the action is also 0, then new direction will be 0 - 1 = -1.
# But it need to be 3. To fix this, I created a function named limit.
def limit(value, minimum=0, maximum=3):
    if value > maximum:
        return minimum
    elif value < minimum:
        return maximum
    else:
        return value

# new_row = old_row + velocity[new_direction]['y']
# new_column = old_column + velocity[new_direction]['x']
velocity = {0: {'x': -1,'y': 0},
            1: {'x': 0, 'y': -1},
            2: {'x': 1, 'y': 0},
            3: {'x': 0, 'y': 1}}

# 0: empty, 1: wall, 2: apple, 3: snake.
color = {0: (0, 200, 100),
         1: (0, 0, 0),
         2: (200, 0, 0),
         3: (35, 35, 35),
         4: (15, 15, 15)}

def integer(_tuple_):
    return tuple(int(element) for element in _tuple_)

def location_of_apple():
    _tuple_ = (apple.row < snake.row,
               apple.row == snake.row,
               apple.row > snake.row,
               apple.column < snake.column,
               apple.column == snake.column,
               apple.column > snake.column)
    return integer(_tuple_)

def correct_moves():
    _tuple_ = location_of_apple()
    _dictionary_ = {(1, 0, 0, 1, 0, 0): (0, 1),
                    (1, 0, 0, 0, 1, 0): (0,),
                    (1, 0, 0, 0, 0, 1): (0, 3),
                    (0, 1, 0, 1, 0, 0): (1,),
                    (0, 1, 0, 0, 1, 0): (None,),
                    (0, 1, 0, 0, 0, 1): (3,),
                    (0, 0, 1, 1, 0, 0): (1, 2),
                    (0, 0, 1, 0, 1, 0): (2,),
                    (0, 0, 1, 0, 0, 1): (2, 3)}
    return _dictionary_[_tuple_]

def check_nearby_cells():
    # Check nearby cells if there is an obstacle or tail.
    row, column = snake.row, snake.column
    print(row, column, map.width, map.height)
    if row == 0:
        _0_ = 1
    else:
        _0_ = map.matrix[row - 1, column] in (1, 3)
        
    if column == 0:
        _1_ = 1
    else:
        _1_ = map.matrix[row, column - 1] in (1, 3)
        
    if row >= (map.width - 1):
        _2_ = 1
    else:
        _2_ = map.matrix[row + 1, column] in (1, 3)
        
    if column >= (map.height - 1):
        _3_ = 1
    else:
        _3_ = map.matrix[row, column + 1] in (1, 3)
        
    _tuple_ = (_0_, _1_, _2_, _3_)
    return integer(_tuple_)

class Cell:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class Text:
    def __init__(self, base_string, function, color=(255, 0, 0)):
        # base_string can be "FPS: " or "Score: " to represent the remaining part.
        self.base_string = base_string
        self.color = color
        # function is used to get the remaining part (It can be fps value or score etc.) of the string.
        # So {base_string + remaining} will be shown on the screen.
        self.function = function
        # Size of the font can be changed.
        self.font = pygame.font.SysFont("Helvetica", 32)

    def show(self, display, position, string=None):
        # If a string is given, use that. Otherwise, use the base_string and add the remaining part to it.
        if string is None:
            string = self.base_string
            remaining = self.function()
            # Check if the type of the remaining part is str. If not, Convert it to the str type before merging strings.
            if type(remaining) is str:
                string += remaining
            else:
                string += str(remaining)
        text = self.font.render(string, True, self.color)
        display.blit(text, position)


class Screen:
    def __init__(self, background_color=(100, 100, 100), resolution=(1000, 750)):
        self.background_color = background_color
        self.width, self.height = resolution
        self.display = pygame.display.set_mode(resolution)
        self.FPS = self.FPS()

    # This functions is used to convert row and column values to x and y values. Or vice versa...
    def convert_position(self, value: int, axis: str):
        if axis == "x":
            result = value * map.cell.width
        elif axis == "y":
            result = value * map.cell.height
        else:
            raise ValueError(f"axis have to be 'x' or 'y' but {axis} is given instead.")
        return result
    
    def draw(self):
        for row in range(map.width):
            for column in range(map.height):
                value = map.matrix[row, column]
                x = self.convert_position(row, axis='x')
                y = self.convert_position(column, axis='y')
                width = map.cell.width
                height = map.cell.height
                geometry = (x, y, width, height)
                pygame.draw.rect(self.display, color[value], geometry)

    def fill(self, color=None):
        if color is None:
            color = self.background_color
        # fill the screen with specific color.
        self.display.fill(color)

    @staticmethod
    def update():
        # Update the whole window.
        pygame.display.flip()

    class FPS:
        def __init__(self):
            self.clock = pygame.time.Clock()
            self.text = Text(base_string="FPS: ", function=self.get)

        def set(self, value):
            self.clock.tick(value)

        def get(self):
            return round(self.clock.get_fps())  # clock.get_fps() returns a float.


class Map:
    def __init__(self, size):
        self.width, self.height = size
        cell_width = screen.width // self.width
        cell_height = screen.height // self.height
        self.cell = Cell(cell_width, cell_height)
        self.matrix = None
        self.create()

    def create(self):
        self.matrix = np.ones((self.width, self.height), dtype=int)
        self.matrix[1: (self.width - 1), 1: (self.height - 1)] = 0

    def random_position(self):
        try:
            empty_cells = np.where(self.matrix == 0)
            index = random.randint(0, len(empty_cells[0]) - 1)
        except ValueError:
            index = 0
        finally:
            cell = empty_cells[0][index], empty_cells[1][index]
        return cell


class Apple:
    def __init__(self):
        self.row, self.column = None, None

    def create(self):
        self.row, self.column = map.random_position()
        map.matrix[self.row, self.column] = 2


class Snake:
    def __init__(self):
        self.row, self.column = None, None
        self.length = 1
        self.tail = []
        self.head = (1, 1)
        self.reward = 0
        self.done = 0
        self.direction = random.randint(0, 3)

    def create(self):
        self.row, self.column = map.random_position()
        self.update_tail()
        self.update_map()

    def reset(self):
        map.matrix[map.matrix == 2] = 0
        map.matrix[map.matrix == 3] = 0
        map.matrix[map.matrix == 4] = 0
        score.value = 0
        self.reward = 0
        self.length = 1
        self.tail = []
        map.create()
        self.create()
        apple.create()
        self.done = 0
        self.direction = random.randint(0, 3)

    def update_direction(self, action):
        if action == 0:
            self.direction = limit(self.direction - 1)
        elif action == 2:
            self.direction = limit(self.direction + 1)

    def update_position(self):
        self.row += velocity[self.direction]['x']
        self.column += velocity[self.direction]['y']

    def update_tail(self):
        self.tail.append((self.row, self.column))
        if len(self.tail) > self.length:
            # Note that, self.tail[-1] gives the position of the head.
            # So, pop the first element instead of the last
            self.tail.pop(0)

    def update_reward(self):
        value = map.matrix[self.row, self.column]
        # First, check if there is a collision with wall or tail.
        if value == 1 or (value == 3 and self.length > 1):
            self.reward = -1000
            self.done = 1
        # If there is no collision, then check if the snake ate the apple.
        elif value == 2:
            self.reward = 1000
            self.grow()
            score.add(1)
        # If didn't, then check if the snake got closer to the apple.
        else:
            if self.direction in correct_moves():
                self.reward = 10
            else:
                self.reward = -10

    def grow(self):
        self.length += 1
        # Check if the snake beated the game.
        if self.length == (map.width - 2) * (map.height - 2):
            print("Successful!")
            self.done = 1
        # If didn't, then create a new apple.
        else:
            apple.create()

    def update_map(self):
        # First, clear the snake from the map.
        map.matrix[map.matrix == 3] = 0
        map.matrix[map.matrix == 4] = 0
        # Now, place the snake again.
        for cell in self.tail[0: (self.length - 1)]:
            row, column = cell
            map.matrix[row, column] = 3
        # Place the head.
        row, column = self.tail[-1]
        map.matrix[row, column] = 4

    def move(self, action):
        self.update_direction(action)
        self.update_position()
        self.update_tail()
        self.update_reward()
        self.update_map()


class Score:
    def __init__(self):
        self.text = Text(base_string="Score: ", function=self.get)
        self.max_value = 0
        self.value = 0

    def add(self, value):
        self.value += value
        if self.value > self.max_value:
            self.max_value = self.value
            print("New Best:", self.max_value)

    def get(self):
        return self.value


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

class Keyboard:
    @staticmethod
    def update():
        global FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # Press 'Q' to quit.
                if event.key == pygame.K_q:
                    print(Q.q_table)
                    pygame.quit()
                    quit()
                if event.key == pygame.K_f:
                    if FPS == 5:
                        FPS = 999999
                    else:
                        FPS = 5
                if event.key == pygame.K_RIGHT:
                    snake.move(action="right")
                elif event.key == pygame.K_LEFT:
                    snake.move(action="left")
                elif event.key == pygame.K_DOWN:
                    if Q.epsilon > 0.1:
                        Q.epsilon = round(Q.epsilon - 0.1, 1)
                    else:
                        Q.epsilon = 0.0
                    print(f"--- Exploration rate: {Q.epsilon}")
                elif event.key == pygame.K_UP:
                    if Q.epsilon < 0.9:
                        Q.epsilon = round(Q.epsilon + 0.1, 1)
                    else:
                        Q.epsilon = 1.0
                    print(f"+++ Exploration rate: {Q.epsilon}")

screen = Screen(resolution=(720, 600))
keyboard = Keyboard()
map = Map(size=(6, 5))
apple = Apple()
snake = Snake()
apple.create()
snake.create()
score = Score()
Q = DQNAgent(state_size=11, action_size=4)
start = datetime.now()
n = 0

while True:
    keyboard.update()
    _state_ = np.array((*check_nearby_cells(), *location_of_apple(), snake.direction), dtype=int)
    _state_ = np.expand_dims(_state_, axis=0)
    print(_state_)
    _action_ = Q.act(_state_)
    snake.move(_action_)
    _reward_ = snake.reward
    _done_ = snake.done
    _new_state_ = np.array((*check_nearby_cells(), *location_of_apple(), snake.direction), dtype=int)
    _new_state_ = np.expand_dims(_new_state_, axis=0)
    Q.remember(_state_, _action_, _reward_, _new_state_, _done_)
    Q.replay()
    if snake.done == 1:
        snake.reset()
    screen.fill()
    screen.draw()
    screen.FPS.text.show(screen.display, position=(0, 0))
    score.text.show(screen.display, position=(200, 0))
    if FPS == 999999:
        if n > 250:
            screen.update()
            n = 0
    else:
        screen.update()
    n += 1
    screen.FPS.set(FPS)
    end = datetime.now()
    difference = end - start
    if difference.total_seconds() > 300:
        start = end
        if Q.epsilon > 0.2:
            Q.epsilon = round(Q.epsilon - 0.1, 1)
        else:
            Q.epsilon = 0.0
        print(f"Exploration rate: {Q.epsilon}")


        


        
