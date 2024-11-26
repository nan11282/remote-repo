import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from collections import deque
import random
import warnings

# 忽略字体警告
warnings.filterwarnings("ignore")

# 设置字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 配置参数
CONFIG = {
    'MAP_WIDTH': 50,  # 地图宽度
    'MAP_HEIGHT': 50,  # 地图高度
    'NUM_CARS': 100,  # 车辆数量
    'NUM_STEPS': 200,  # 仿真步数
    'MAX_SPEED': 1,  # 车辆最大速度
    'COMMUNITIES': [  # 小区简化为整体区域，并确保与主干道相连
        {'x': 10, 'y': 5, 'width': 10, 'height': 10},
        {'x': 30, 'y': 5, 'width': 10, 'height': 10},
        {'x': 10, 'y': 35, 'width': 10, 'height': 10},
        {'x': 30, 'y': 35, 'width': 10, 'height': 10},
    ]
}

# 地图元素标记
ROAD = 0
BUILDING = 1
COMMUNITY = 2
CAR = 3  # 表示车辆的位置

# 颜色映射
COLOR_MAP = {
    ROAD: 'gray',       # 道路颜色
    BUILDING: 'black',  # 建筑物颜色
    COMMUNITY: 'green', # 小区区域
    CAR: 'red'          # 车辆颜色
}

def create_map():
    # 初始化地图，全为建筑物
    grid_map = np.ones((CONFIG['MAP_HEIGHT'], CONFIG['MAP_WIDTH']), dtype=int) * BUILDING

    # 设置主干道（两条横向主干道和两条垂直主干道）
    grid_map[CONFIG['MAP_HEIGHT'] // 4, :] = ROAD
    grid_map[3 * CONFIG['MAP_HEIGHT'] // 4, :] = ROAD
    grid_map[:, CONFIG['MAP_WIDTH'] // 4] = ROAD
    grid_map[:, 3 * CONFIG['MAP_WIDTH'] // 4] = ROAD

    # 设置小区并与主干道相连
    for community in CONFIG['COMMUNITIES']:
        x = community['x']
        y = community['y']
        width = community['width']
        height = community['height']
        # 将整个小区标记为COMMUNITY
        grid_map[y:y+height, x:x+width] = COMMUNITY
        # 确保小区直接与主干道相连
        connect_community_to_road(grid_map, x, y, width, height)

    return grid_map

def connect_community_to_road(grid_map, x, y, width, height):
    # 确保小区边缘与主干道连接
    # 上边界连接
    if y > 0:
        grid_map[y - 1, x:x+width] = ROAD
    # 下边界连接
    if y + height < CONFIG['MAP_HEIGHT']:
        grid_map[y + height, x:x+width] = ROAD
    # 左边界连接
    if x > 0:
        grid_map[y:y+height, x - 1] = ROAD
    # 右边界连接
    if x + width < CONFIG['MAP_WIDTH']:
        grid_map[y:y+height, x + width] = ROAD

class Car:
    def __init__(self, start_pos):
        self.position = start_pos
        self.path = deque()
        self.travel_time = 0
        self.arrived = False

    def plan_route(self, grid_map):
        # 简单的随机路径规划，寻找一个目标（例如建筑物）
        queue = deque([(self.position, [])])
        visited = set([self.position])

        while queue:
            current_pos, path = queue.popleft()
            x, y = current_pos

            if grid_map[y, x] == BUILDING:  # 到达建筑物区域
                self.path = deque(path)
                return

            # 检查邻居节点
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < CONFIG['MAP_WIDTH'] and 0 <= ny < CONFIG['MAP_HEIGHT']:
                    if (nx, ny) not in visited and grid_map[ny, nx] in [ROAD, COMMUNITY]:
                        queue.append(((nx, ny), path + [(nx, ny)]))
                        visited.add((nx, ny))

        # 如果没有找到路径
        if not self.path:
            print(f"车辆无法规划路径，从起点 {self.position} 到达目标！")

def init_cars(grid_map):
    cars = []
    for _ in range(CONFIG['NUM_CARS']):
        # 随机选择主干道上的起点
        while True:
            start_pos = (random.randint(0, CONFIG['MAP_WIDTH'] - 1), random.randint(0, CONFIG['MAP_HEIGHT'] - 1))
            if grid_map[start_pos[1], start_pos[0]] == ROAD:
                break
        car = Car(start_pos)
        car.plan_route(grid_map)
        if car.path:
            cars.append(car)
    return cars

def simulate(grid_map, cars):
    map_states = []

    for step in range(CONFIG['NUM_STEPS']):
        car_map = np.full((CONFIG['MAP_HEIGHT'], CONFIG['MAP_WIDTH']), -1)

        for car in cars:
            if car.arrived or not car.path:
                continue

            next_pos = car.path.popleft()
            car.position = next_pos
            car.travel_time += 1
            if grid_map[next_pos[1], next_pos[0]] == BUILDING:
                car.arrived = True

            car_map[car.position[1], car.position[0]] = CAR

        combined_map = grid_map.copy()
        combined_map[car_map == CAR] = CAR
        map_states.append(combined_map)

    return map_states

def analyze_simulation(cars, total_steps):
    # 到达目标的车辆
    arrived_cars = [car for car in cars if car.arrived]
    # 计算平均行程时间
    avg_travel_time = sum(car.travel_time for car in arrived_cars) / len(arrived_cars) if arrived_cars else 0
    # 总车辆数量与到达目标车辆的比例
    total_cars = len(cars)
    arrived_ratio = len(arrived_cars) / total_cars if total_cars > 0 else 0

    print("仿真结果分析:")
    print(f"总车辆数量: {total_cars}")
    print(f"到达目的地的车辆数量: {len(arrived_cars)}")
    print(f"平均行程时间: {avg_travel_time:.2f} 步")
    print(f"到达比例: {arrived_ratio:.2%}")
    print(f"总步数: {total_steps}")

def run_simulation():
    grid_map = create_map()
    cars = init_cars(grid_map)
    map_states = simulate(grid_map, cars)

    cmap = ListedColormap([COLOR_MAP[ROAD], COLOR_MAP[BUILDING], COLOR_MAP[COMMUNITY], COLOR_MAP[CAR]])

    fig, ax = plt.subplots(figsize=(8, 8))
    ims = []

    for state in map_states:
        im = ax.imshow(state, animated=True, cmap=cmap, vmin=0, vmax=3)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    plt.show()

    # 分析仿真结果
    analyze_simulation(cars, CONFIG['NUM_STEPS'])
