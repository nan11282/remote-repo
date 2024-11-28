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

# 地图元素标记
EMPTY = 0
MAIN_ROAD = 1
COMMUNITY = 2
CAR = 3  # 表示车辆的位置

# 颜色映射
COLOR_MAP = {
    EMPTY: 'white',
    MAIN_ROAD: 'gray',
    COMMUNITY: 'lightgreen',
    CAR: 'red'
}

# 方向定义
DIRECTIONS = ['N', 'S', 'E', 'W']
# 修正后的方向移动定义
DX = {'N': 0, 'S': 0, 'E': 1, 'W': -1}
DY = {'N': 1, 'S': -1, 'E': 0, 'W': 0}

class Car:
    def __init__(self, car_id, entry, exit_dir, speed):
        self.id = car_id
        self.entry = entry            # 进入方向
        self.exit_dir = exit_dir      # 退出方向
        self.position = entry_position(entry)
        self.destination = exit_position(exit_dir)
        self.path = deque()
        self.arrived = False
        self.speed = speed
        self.travel_time = 0
        self.take_community = False  # 是否选择走小区道路

    def plan_route(self, grid_map, main_road_density, community_density, communities_open):
        """
        根据主干道和小区道路的交通密度，选择是否走小区道路。
        """
        if not communities_open:
            self.take_community = False
        else:
            # 决策逻辑：
            # 如果主干道密度 > 阈值，且小区道路密度 < 阈值，选择走小区道路
            if main_road_density > CONFIG['THRESHOLD_MAIN'] and community_density < CONFIG['THRESHOLD_COMM']:
                self.take_community = True
            else:
                self.take_community = False

        # 基本路径规划：直接走主干道或通过小区道路
        if self.take_community:
            self.path = calculate_path_via_community(self.entry, self.exit_dir, grid_map)
        else:
            self.path = calculate_direct_path(self.entry, self.exit_dir, grid_map)

def entry_position(direction):
    """
    根据进入方向，返回车辆的起始位置。
    """
    mid = CONFIG['MAP_SIZE'] // 2
    if direction == 'N':
        return (mid, 0)
    elif direction == 'S':
        return (mid, CONFIG['MAP_SIZE'] - 1)
    elif direction == 'E':
        return (CONFIG['MAP_SIZE'] -1, mid)
    elif direction == 'W':
        return (0, mid)

def exit_position(direction):
    """
    根据退出方向，返回车辆的目标位置。
    """
    mid = CONFIG['MAP_SIZE'] // 2
    if direction == 'N':
        return (mid, 0)
    elif direction == 'S':
        return (mid, CONFIG['MAP_SIZE'] - 1)
    elif direction == 'E':
        return (CONFIG['MAP_SIZE'] -1, mid)
    elif direction == 'W':
        return (0, mid)

def calculate_direct_path(entry, exit_dir, grid_map):
    """
    计算直接从入口到出口的路径（不走小区道路）。
    """
    path = deque()
    x, y = entry_position(entry)
    target_x, target_y = exit_position(exit_dir)

    mid = CONFIG['MAP_SIZE']//2

    # 移动到十字路口
    while (x, y) != (mid, mid):
        if x < mid:
            x += 1
        elif x > mid:
            x -= 1
        if y < mid:
            y += 1
        elif y > mid:
            y -= 1
        # 添加边界检查
        if 0 <= x < CONFIG['MAP_SIZE'] and 0 <= y < CONFIG['MAP_SIZE']:
            path.append((x, y))
        else:
            break  # 防止超出边界

    # 从十字路口移动到出口
    while (x, y) != (target_x, target_y):
        if x < target_x:
            x += 1
        elif x > target_x:
            x -= 1
        if y < target_y:
            y += 1
        elif y > target_y:
            y -= 1
        # 添加边界检查
        if 0 <= x < CONFIG['MAP_SIZE'] and 0 <= y < CONFIG['MAP_SIZE']:
            path.append((x, y))
        else:
            break  # 防止超出边界

    return path

def calculate_path_via_community(entry, exit_dir, grid_map):
    """
    计算通过小区道路的路径。
    """
    path = deque()
    x, y = entry_position(entry)
    target_x, target_y = exit_position(exit_dir)

    mid = CONFIG['MAP_SIZE'] // 2
    comm_width = CONFIG['COMMUNITY_WIDTH']

    # 进入小区道路
    if entry in ['N', 'S']:
        # 从北或南进入，选择左或右进入小区道路
        if exit_dir == 'E':
            # 右转进入小区道路，向东移动
            for _ in range(comm_width):
                x +=1
                if 0 <= x < CONFIG['MAP_SIZE']:
                    path.append((x, y))
                else:
                    break
        elif exit_dir == 'W':
            # 左转进入小区道路，向西移动
            for _ in range(comm_width):
                x -=1
                if 0 <= x < CONFIG['MAP_SIZE']:
                    path.append((x, y))
                else:
                    break
    elif entry in ['E', 'W']:
        # 从东或西进入，选择上或下进入小区道路
        if exit_dir == 'N':
            # 右转进入小区道路，向北移动
            for _ in range(comm_width):
                y -=1
                if 0 <= y < CONFIG['MAP_SIZE']:
                    path.append((x, y))
                else:
                    break
        elif exit_dir == 'S':
            # 左转进入小区道路，向南移动
            for _ in range(comm_width):
                y +=1
                if 0 <= y < CONFIG['MAP_SIZE']:
                    path.append((x, y))
                else:
                    break

    # 从小区道路返回主干道，向出口方向移动
    while (x, y) != (target_x, target_y):
        if x < target_x:
            x +=1
        elif x > target_x:
            x -=1
        if y < target_y:
            y +=1
        elif y > target_y:
            y -=1
        # 添加边界检查
        if 0 <= x < CONFIG['MAP_SIZE'] and 0 <= y < CONFIG['MAP_SIZE']:
            path.append((x, y))
        else:
            break

    return path

def create_map(communities_open=True):
    """
    创建地图，包括主干道和小区道路。
    """
    grid_map = np.zeros((CONFIG['MAP_SIZE'], CONFIG['MAP_SIZE']), dtype=int)

    mid = CONFIG['MAP_SIZE'] // 2

    # 设置主干道
    grid_map[mid, :] = MAIN_ROAD
    grid_map[:, mid] = MAIN_ROAD

    if communities_open:
        # 设置小区道路（四个方向）
        for i in range(1, CONFIG['COMMUNITY_WIDTH'] +1):
            # 北小区道路
            grid_map[mid - i, mid - CONFIG['COMMUNITY_WIDTH']] = COMMUNITY
            grid_map[mid - i, mid + CONFIG['COMMUNITY_WIDTH']] = COMMUNITY
            # 南小区道路
            grid_map[mid + i, mid - CONFIG['COMMUNITY_WIDTH']] = COMMUNITY
            grid_map[mid + i, mid + CONFIG['COMMUNITY_WIDTH']] = COMMUNITY
            # 东小区道路
            grid_map[mid - CONFIG['COMMUNITY_WIDTH'], mid + i] = COMMUNITY
            grid_map[mid + CONFIG['COMMUNITY_WIDTH'], mid + i] = COMMUNITY
            # 西小区道路
            grid_map[mid - CONFIG['COMMUNITY_WIDTH'], mid - i] = COMMUNITY
            grid_map[mid + CONFIG['COMMUNITY_WIDTH'], mid - i] = COMMUNITY
    else:
        # 小区道路关闭，设置为 EMPTY
        for i in range(1, CONFIG['COMMUNITY_WIDTH'] +1):
            # 北小区道路
            grid_map[mid - i, mid - CONFIG['COMMUNITY_WIDTH']] = EMPTY
            grid_map[mid - i, mid + CONFIG['COMMUNITY_WIDTH']] = EMPTY
            # 南小区道路
            grid_map[mid + i, mid - CONFIG['COMMUNITY_WIDTH']] = EMPTY
            grid_map[mid + i, mid + CONFIG['COMMUNITY_WIDTH']] = EMPTY
            # 东小区道路
            grid_map[mid - CONFIG['COMMUNITY_WIDTH'], mid + i] = EMPTY
            grid_map[mid + CONFIG['COMMUNITY_WIDTH'], mid + i] = EMPTY
            # 西小区道路
            grid_map[mid - CONFIG['COMMUNITY_WIDTH'], mid - i] = EMPTY
            grid_map[mid + CONFIG['COMMUNITY_WIDTH'], mid - i] = EMPTY

    return grid_map

def init_cars():
    """
    初始化车辆列表为空，车辆将在仿真过程中动态生成。
    """
    cars = []
    car_id = 0
    return cars, car_id

def is_main_road(position):
    """
    判断当前位置是否在主干道。
    """
    mid = CONFIG['MAP_SIZE'] // 2
    x, y = position
    return x == mid or y == mid

def is_community(position):
    """
    判断当前位置是否在小区道路。
    """
    mid = CONFIG['MAP_SIZE'] // 2
    comm_width = CONFIG['COMMUNITY_WIDTH']
    x, y = position
    # 小区道路范围，排除主干道
    return ((mid - comm_width <= x <= mid + comm_width) or
            (mid - comm_width <= y <= mid + comm_width)) and not is_main_road(position)

def simulate_step(grid_map, cars, car_id_counter, communities_open):
    """
    执行一个仿真步，并返回当前地图状态。
    """
    # 生成新车
    for direction in DIRECTIONS:
        if random.random() < CONFIG['ARRIVAL_RATES'][direction]:
            # 选择一个随机的出口方向
            exit_choices = [d for d in DIRECTIONS if d != direction]
            exit_dir = random.choice(exit_choices)
            # 随机分配速度
            speed = random.randint(1, 3)  # 速度范围可以根据需要调整
            # 创建新车
            new_car = Car(car_id_counter, direction, exit_dir, speed)
            # 规划路径
            # 在规划路径前，统计主干道和小区道路的车辆密度
            main_road_density = sum(1 for car in cars if is_main_road(car.position))
            community_density = sum(1 for car in cars if is_community(car.position))
            new_car.plan_route(grid_map, main_road_density, community_density, communities_open)
            if new_car.path:
                cars.append(new_car)
            car_id_counter +=1

    # 统计主干道和小区道路上的车辆密度
    main_road_density = sum(1 for car in cars if is_main_road(car.position))
    community_density = sum(1 for car in cars if is_community(car.position))

    # 动态调整主干道速度
    # 主干道速度随着主干道车辆密度增加而减慢
    main_speed = max(1, int(CONFIG['MAIN_ROAD_BASE_SPEED'] - CONFIG['MAIN_ROAD_SPEED_DECREMENT'] * main_road_density))

    # Initialize occupancy map
    occupancy = np.zeros_like(grid_map)

    # Move cars
    for car in cars:
        if car.arrived:
            continue
        # Determine the number of cells to move based on speed
        current_speed = main_speed if is_main_road(car.position) else CONFIG['COMMUNITY_SPEED']
        for _ in range(current_speed):
            if not car.path:
                break
            next_pos = car.path.popleft()
            nx, ny = next_pos
            # Check if next_pos is within bounds
            if 0 <= nx < CONFIG['MAP_SIZE'] and 0 <= ny < CONFIG['MAP_SIZE']:
                # Check if next_pos is already occupied
                if occupancy[ny, nx] == 0:
                    # Move car
                    car.position = (nx, ny)
                    car.travel_time +=1
                    # Mark as occupied
                    occupancy[ny, nx] = 1
                    # Check if arrived
                    if (nx, ny) == car.destination:
                        car.arrived = True
                        break
                else:
                    # Cannot move, wait
                    break
            else:
                # Out of bounds, mark as arrived
                car.arrived = True
                break

    # Update map
    current_map = grid_map.copy()
    for car in cars:
        if not car.arrived:
            x, y = car.position
            if 0 <= x < CONFIG['MAP_SIZE'] and 0 <= y < CONFIG['MAP_SIZE']:
                current_map[y, x] = CAR

    return current_map, car_id_counter

def analyze_simulation(cars, total_steps, scenario):
    """
    分析仿真结果，并打印关键数据。
    """
    arrived_cars = [car for car in cars if car.arrived]
    avg_travel_time = sum(car.travel_time for car in arrived_cars) / len(arrived_cars) if arrived_cars else 0
    total_cars = len(cars)
    arrived_ratio = len(arrived_cars) / total_cars if total_cars >0 else 0

    print(f"=== {scenario} ===")
    print("仿真结果分析:")
    print(f"总车辆数量: {total_cars}")
    print(f"到达目的地的车辆数量: {len(arrived_cars)}")
    print(f"平均行程时间: {avg_travel_time:.2f} 步")
    print(f"到达比例: {arrived_ratio:.2%}")
    print(f"总步数: {total_steps}")
    print("-"*30)

def create_animation(grid_map, cars, car_id_counter, communities_open, scenario_label):
    """
    创建动画，同时进行仿真和数据收集。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = ListedColormap([COLOR_MAP[EMPTY], COLOR_MAP[MAIN_ROAD], COLOR_MAP[COMMUNITY], COLOR_MAP[CAR]])
    im = ax.imshow(grid_map, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(scenario_label)
    plt.axis('off')

    def update(frame):
        nonlocal cars, car_id_counter
        if frame >= CONFIG['NUM_STEPS']:
            return [im]
        current_map, car_id_counter = simulate_step(grid_map, cars, car_id_counter, communities_open)
        im.set_data(current_map)
        return [im]

    def on_animation_end(*args):
        analyze_simulation(cars, CONFIG['NUM_STEPS'], scenario_label)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=CONFIG['NUM_STEPS'],
        interval=100,
        blit=True,
        repeat=False
    )

    # 连接动画结束的事件
    def handle_close(evt):
        on_animation_end()

    fig.canvas.mpl_connect('close_event', handle_close)

    plt.show()

def main():
    """
    主函数，运行单一仿真并可视化。
    """
    print("欢迎使用交通仿真系统！请根据提示输入仿真参数。")

    # 获取用户输入的仿真参数
    try:
        map_size = int(input("请输入地图大小（建议为奇数，如21）："))
        if map_size % 2 == 0:
            print("地图大小应为奇数，自动加1。")
            map_size +=1
    except:
        print("输入无效，使用默认地图大小21。")
        map_size = 21

    try:
        community_width = int(input("请输入小区道路宽度（例如3）："))
    except:
        print("输入无效，使用默认小区道路宽度3。")
        community_width = 3

    try:
        num_steps = int(input("请输入仿真步数（例如200）："))
    except:
        print("输入无效，使用默认仿真步数200。")
        num_steps = 200

    try:
        threshold_main = int(input("请输入主干道密度阈值（例如5）："))
    except:
        print("输入无效，使用默认主干道密度阈值5。")
        threshold_main = 5

    try:
        threshold_comm = int(input("请输入小区道路密度阈值（例如5）："))
    except:
        print("输入无效，使用默认小区道路密度阈值5。")
        threshold_comm = 5

    try:
        main_road_base_speed = float(input("请输入主干道基础速度（步/步，例如3）："))
    except:
        print("输入无效，使用默认主干道基础速度3。")
        main_road_base_speed = 3

    try:
        main_road_speed_decrement = float(input("请输入主干道速度随密度增加的减速度（例如0.1）："))
    except:
        print("输入无效，使用默认主干道速度减速度0.1。")
        main_road_speed_decrement = 0.1

    try:
        community_speed = int(input("请输入小区道路速度（步/步，例如2）："))
    except:
        print("输入无效，使用默认小区道路速度2。")
        community_speed = 2

    # 设置ARRIVAL_RATES，默认四个方向相同
    try:
        arrival_rate_n = float(input("请输入北方向车辆到达率（每步概率，0到1，例如0.2）："))
        arrival_rate_s = float(input("请输入南方向车辆到达率（每步概率，0到1，例如0.2）："))
        arrival_rate_e = float(input("请输入东方向车辆到达率（每步概率，0到1，例如0.2）："))
        arrival_rate_w = float(input("请输入西方向车辆到达率（每步概率，0到1，例如0.2）："))
        arrival_rates = {
            'N': arrival_rate_n,
            'S': arrival_rate_s,
            'E': arrival_rate_e,
            'W': arrival_rate_w
        }
    except:
        print("输入无效，使用默认四个方向到达率0.2。")
        arrival_rates = {
            'N': 0.2,
            'S': 0.2,
            'E': 0.2,
            'W': 0.2
        }

    # 选择小区道路是否开放
    try:
        communities_open_input = input("是否开放小区道路？（y/n，默认y）：").lower()
        communities_open = True if communities_open_input != 'n' else False
    except:
        communities_open = True

    # 更新CONFIG
    global CONFIG
    CONFIG = {
        'MAP_SIZE': map_size,               # 地图大小
        'COMMUNITY_WIDTH': community_width, # 小区道路宽度
        'ARRIVAL_RATES': arrival_rates,      # 各方向车辆到达率
        'MAIN_ROAD_BASE_SPEED': main_road_base_speed,    # 主干道基础速度
        'MAIN_ROAD_SPEED_DECREMENT': main_road_speed_decrement,  # 主干道速度减量
        'COMMUNITY_SPEED': community_speed,  # 小区道路速度
        'NUM_STEPS': num_steps,              # 仿真步数
        'THRESHOLD_MAIN': threshold_main,    # 主干道密度阈值
        'THRESHOLD_COMM': threshold_comm     # 小区道路密度阈值
    }

    # 创建地图
    grid_map = create_map(communities_open)

    # 初始化车辆列表
    cars, car_id_counter = init_cars()

    # 设置仿真场景标签
    scenario_label = "小区开放" if communities_open else "小区关闭"

    print(f"开始仿真：{scenario_label}，步数：{CONFIG['NUM_STEPS']}")

    # 创建并运行动画
    create_animation(grid_map, cars, car_id_counter, communities_open, scenario_label)

if __name__ == '__main__':
    main()