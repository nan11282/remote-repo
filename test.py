import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import random

# 地图尺寸
MAP_WIDTH = 50
MAP_HEIGHT = 50

# 车辆数量
NUM_CARS = 100

# 仿真步数
NUM_STEPS = 100

# 车辆最大速度
MAX_SPEED = 1  # 在网格中每步只能移动一格

# 小区位置和大小
COMMUNITY_X = 20
COMMUNITY_Y = 20
COMMUNITY_WIDTH = 10
COMMUNITY_HEIGHT = 10

# 道路标记
ROAD = 0
BUILDING = 1
COMMUNITY_ROAD = 2

# 小区是否开放
COMMUNITY_OPEN = False  # False表示小区封闭，True表示小区开放

def create_map():
    # 初始化地图，全为建筑物
    grid_map = np.ones((MAP_HEIGHT, MAP_WIDTH)) * BUILDING

    # 设置主干道（横纵各一条）
    grid_map[MAP_HEIGHT // 2, :] = ROAD
    grid_map[:, MAP_WIDTH // 2] = ROAD

    # 设置小区内部道路（如果小区开放）
    if COMMUNITY_OPEN:
        grid_map[COMMUNITY_Y:COMMUNITY_Y + COMMUNITY_HEIGHT,
                 COMMUNITY_X:COMMUNITY_X + COMMUNITY_WIDTH] = COMMUNITY_ROAD

        # 连接小区内部道路和主干道
        grid_map[COMMUNITY_Y + COMMUNITY_HEIGHT // 2,
                 COMMUNITY_X - 1] = ROAD
        grid_map[COMMUNITY_Y - 1,
                 COMMUNITY_X + COMMUNITY_WIDTH // 2] = ROAD

    return grid_map

class Car:
    def __init__(self, start_pos, end_pos):
        self.position = start_pos  # 当前位置
        self.end_pos = end_pos  # 目的地
        self.path = deque()  # 路径
        self.speed = MAX_SPEED  # 速度
        self.travel_time = 0  # 行程时间
        self.arrived = False  # 是否到达目的地

    def plan_route(self, grid_map):
        # 使用简单的曼哈顿距离规划路径
        x0, y0 = self.position
        x1, y1 = self.end_pos
        path = []

        # 水平移动
        dx = 1 if x1 > x0 else -1
        for x in range(x0, x1, dx):
            if grid_map[y0, x] != BUILDING:
                path.append((x, y0))
            else:
                break  # 遇到障碍物

        # 垂直移动
        dy = 1 if y1 > y0 else -1
        for y in range(y0, y1, dy):
            if grid_map[y, x1] != BUILDING:
                path.append((x1, y))
            else:
                break  # 遇到障碍物

        self.path = deque(path)

def init_cars(grid_map):
    cars = []

    # 随机生成车辆的起点和终点
    for _ in range(NUM_CARS):
        while True:
            # 起点
            x_start = random.randint(0, MAP_WIDTH - 1)
            y_start = random.randint(0, MAP_HEIGHT - 1)
            if grid_map[y_start, x_start] == ROAD:
                break

        while True:
            # 终点
            x_end = random.randint(0, MAP_WIDTH - 1)
            y_end = random.randint(0, MAP_HEIGHT - 1)
            if grid_map[y_end, x_end] == ROAD or (COMMUNITY_OPEN and grid_map[y_end, x_end] == COMMUNITY_ROAD):
                break

        car = Car(start_pos=(x_start, y_start), end_pos=(x_end, y_end))
        car.plan_route(grid_map)
        cars.append(car)

    return cars

def simulate(grid_map, cars):
    # 用于可视化
    map_states = []

    for step in range(NUM_STEPS):
        map_state = grid_map.copy()

        # 创建位置到车辆的映射，避免碰撞
        pos_to_car = {}

        for car in cars:
            if not car.arrived:
                # 如果有预先规划的路径，则按照路径移动
                if car.path:
                    next_pos = car.path.popleft()
                    if next_pos not in pos_to_car:
                        pos_to_car[next_pos] = car
                        car.position = next_pos
                    else:
                        # 前方位置被占用，车辆停留在原地
                        car.path.appendleft(next_pos)
                else:
                    # 已到达目的地
                    car.arrived = True

                car.travel_time += 1

            x, y = car.position
            map_state[y, x] = 0.5  # 用0.5表示车辆位置

        map_states.append(map_state)

    return map_states

def run_simulation():
    grid_map = create_map()
    cars = init_cars(grid_map)
    map_states = simulate(grid_map, cars)

    # 可视化
    fig, ax = plt.subplots()
    ims = []

    for state in map_states:
        im = ax.imshow(state, animated=True, cmap='gray', vmin=0, vmax=1)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    plt.show()

    # 计算平均行程时间
    total_time = sum(car.travel_time for car in cars)
    avg_travel_time = total_time / NUM_CARS
    print(f"平均行程时间: {avg_travel_time} 步")

    # 统计到达目的地的车辆数量
    arrived_cars = sum(1 for car in cars if car.arrived)
    print(f"到达目的地的车辆数: {arrived_cars} / {NUM_CARS}")

if __name__ == "__main__":
    # 模拟小区封闭状态
    COMMUNITY_OPEN = False
    print("小区封闭状态:")
    run_simulation()

    # 模拟小区开放状态
    COMMUNITY_OPEN = True
    print("\n小区开放状态:")
    run_simulation()