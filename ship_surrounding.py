# ship_surrounding.py
import time  # 導入 time 庫，用於時間相關操作
import threading  # 導入 threading 庫，用於多線程
import math  # 導入 math 庫，用於數學計算
from ship_navigation_v1 import multi_ship_planning, filter_waypoints_improved, plot_all_ships_paths
import os
# 這行一定要在任何可能載入 MKL / Fortran runtime 之前執行
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "TRUE"

import copy
import config

from objects_collide_control import collide_avoidance_thread, bubble_list_update_thread, draw_bubbles_thread
from Missionplanner_utils import BoatController, MessageMonitor
from formation_set import all_follow_leader

# =========================
# ★ 全域參數
# =========================

stop_flag           = False
boat_num            = 5
start_surround_r    = 40        # 進入此半徑才開始計算包圍 (m)
surrounding_r       = 20         # 真正包圍半徑 (m)
obstacle_r          = 10         # 敵船禁入半徑 (m)

arrival_flags       = [False]*boat_num   # 每艘是否已到 start_surround_r
planning_done       = False
planning_lock       = threading.Lock()
planning_result_xy  = {}          # ShipX → {"path":[(x,y)…]}

boats               = []          # BoatController 物件
# boats_data          = []          # 規劃用 (id,pos,goal)
global_origin       = None        # (lat0, lon0) 供 latlon↔local
finished            = 0           # 完成路徑計算的船數
first_plot_done     = False
stop_follow         = False

ENEMY_OFFSET        = 80

# =========================
# ★ 地理座標 ↔ 平面座標
# =========================

class Enemy_boat:
    def __init__(self, lat, lon, r):
        self.lat = lat
        self.lon = lon
        self.radius = r


def global_to_planning(lat, lon, origin):
    """WGS‑84 → 簡易 local‑xy (m) (正東 +x, 正北 +y)"""
    R = 6371000.0
    lat0, lon0 = origin
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * R * math.cos(math.radians(lat0))
    y = dlat * R
    return (x, y)

def planning_to_global(x, y, origin):
    """local‑xy → WGS‑84"""
    R = 6371000.0
    lat0, lon0 = origin
    lat = lat0 + math.degrees(y / R)
    lon = lon0 + math.degrees(x / (R*math.cos(math.radians(lat0))))
    return lat, lon

def generate_surrounding_targets(center_lat, center_lon, radius_m, n):
    R = 6371000.0
    lat0 = math.radians(center_lat)
    lon0 = math.radians(center_lon)
    pts = []
    for i in range(n):
        theta = -(2*math.pi*i)/n + 0.5 * math.pi
        dlat = radius_m/R * math.cos(theta)
        dlon = radius_m/(R*math.cos(lat0)) * math.sin(theta)
        pts.append((math.degrees(lat0+dlat), math.degrees(lon0+dlon)))
    return pts

def telemetry_printer(enemy, interval=5):
    """每 interval 秒列印各船座標與距離敵船 (m)"""
    while not stop_flag and not planning_done:
        print("─" * 60)
        for b in boats:
            d = b.calculate_distance(
                b.msg_monitor.current_lat, b.msg_monitor.current_lon,
                enemy.lat, enemy.lon)
            print(f"Boat {b.boat_id:>2}: "
                  f"({b.msg_monitor.current_lat:.7f}, "
                  f"{b.msg_monitor.current_lon:.7f})  "
                  f"→ 距敵船 {d:7.1f} m"
                  f"  速度(groundspeed) = {b.msg_monitor.groundspeed:.1f}")
        time.sleep(interval)  

def enemy_keep_moving(enemy, *, speed=0.1, dt=0.02):
    """
    讓虛擬敵船以 0.3 m/s 在方形路徑上來回晃動。
    更新頻率 0.02 s，距離誤差 < 1 cm。
    不會改動 Enemy_boat 既有屬性，只額外掛 _origin / _local / _seg / _prog。
    """
    # 八條 10 m 段向量：左→右→上→下→右→左→下→上
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1),
                  (1, 0), (-1, 0), (0, -1), (0, 1)]
    seg_len = 10.0                   # 每段 10 m
    step_len = speed * dt            # 一步 0.006 m

    # ----- lazy 初始化 -----
    if not hasattr(enemy, "_origin"):
        enemy._origin = (enemy.lat, enemy.lon)   # 起點 (lat, lon)
        enemy._local  = [0.0, 0.0]               # 當前平面座標 (x, y) [m]
        enemy._seg    = 0                        # 目前走到第幾段
        enemy._prog   = 0.0                      # 這段已移動距離 [m]

    while not stop_flag:
        # 目前方向單位向量
        dx, dy = directions[enemy._seg]

        # 推進一步（保持不超過 seg_len）
        move = min(step_len, seg_len - enemy._prog)
        enemy._local[0] += dx * move
        enemy._local[1] += dy * move
        enemy._prog     += move

        # 段落完成 → 換下一段
        if abs(enemy._prog - seg_len) < 1e-6:
            enemy._seg  = (enemy._seg + 1) % len(directions)
            enemy._prog = 0.0  # 重置本段累積距離
            print(f"敵船位置：{enemy.lat}, {enemy.lon}")

        # 由 local-xy 轉回經緯度，寫回敵船位置
        enemy.lat, enemy.lon = planning_to_global(
            enemy._local[0], enemy._local[1], enemy._origin
        )

        time.sleep(dt)

def cal_surrounding_path(enemy_boat):
    global planning_done, planning_result_xy, finished, first_plot_done

    enemy_lat = enemy_boat.lat
    enemy_lon = enemy_boat.lon
    formation = generate_surrounding_targets(enemy_lat, enemy_lon, surrounding_r, boat_num)

    ships_data = []
    for i, b in enumerate(boats):
        x, y = global_to_planning(b.msg_monitor.current_lat, b.msg_monitor.current_lon, global_origin)
        gx, gy = global_to_planning(formation[i][0], formation[i][1], global_origin)
        ships_data.append({
            "id": f"Ship{i+1}",
            "pos": (x,y),
            "goal": (gx, gy)
        })

    obstacles = [(
        *global_to_planning(enemy_lat, enemy_lon, global_origin),
        obstacle_r
    )]


    raw_result = multi_ship_planning(
        ships_data,
        safe_distance=3,
        grid_scale=1,
        smoothing_method="moving_average",
        obstacles=obstacles
    )

    result = copy.deepcopy(raw_result)

    for i, b in enumerate(boats):
        ship_id = f"Ship{i+1}"
        raw_path = result[ship_id]["path"]
        start_xy = ships_data[i]["pos"]

        filtered_path, _ = filter_waypoints_improved(
            raw_path,
            min_distance=8,
            turn_simplify_factor=2.0,
            start_pos=start_xy,
            min_start_distance=15
        )

        result[ship_id]["path"] = filtered_path

        b.current_path = [planning_to_global(x, y, global_origin)
                            for (x, y) in filtered_path]
        
        b.total_distance = sum(
            max(math.hypot(filtered_path[j+1][0] - filtered_path[j][0],filtered_path[j+1][1] - filtered_path[j][1]),0.01)
            for j in range(len(filtered_path)-1)
        )

    print("\n--- 路徑比較（濾波前 vs. 濾波後）---")
    for ship_id in result.keys():
        before = raw_result[ship_id]["path"]   # 濾波前
        after  = result[ship_id]["path"]       # 濾波後

        print(f"{ship_id}:")
        print(f"  濾波前，共 {len(before)} 點")
        for idx, (x, y) in enumerate(before):
            print(f"    前{idx:02d}: ({x:.2f}, {y:.2f})")

        print(f"  濾波後，共 {len(after)} 點")
        for idx, (x, y) in enumerate(after):
            print(f"    後{idx:02d}: ({x:.2f}, {y:.2f})")

        print("-" * 40)


    if not first_plot_done:
        enemy_xy = global_to_planning(enemy_lat, enemy_lon, global_origin)
        out_file = f"OutPhoto/surround_enemy_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_all_ships_paths(ships_data, result, "circle", surrounding_r, *enemy_xy, out_file)
        print("Path figure saved to", out_file)
        first_plot_done = True

    planning_result_xy = result
    planning_done = True

def watcher_thread(enemy_boat):
    global arrival_flags, planning_done, stop_follow
    b1_done = False

    while not stop_flag:
        near_b1 = True
        b1 = boats[0]
        dist = b1.calculate_distance(
            b1.msg_monitor.current_lat, b1.msg_monitor.current_lon,
            enemy_boat.lat, enemy_boat.lon
        )
        if dist <= start_surround_r:
            print(f"領頭船到達包圍半徑")
            stop_follow = True
            b1_done = True
            b1.set_position_target(b1.msg_monitor.current_lat, b1.msg_monitor.current_lon)
            b1.disarm()
            b1.set_mode("HOLD")

        for b in boats:
            if b == b1:
                continue
            if hasattr(b, "last_target"):
                dist_i = b.calculate_distance(
                    b.msg_monitor.current_lat, b.msg_monitor.current_lon,
                    b.last_target[0], b.last_target[1]
                )
                if dist_i > 10:
                    near_b1 = False
            else:
                near_b1 = True
        if near_b1 and b1_done:
            break
        time.sleep(0.2)

    b1 = boats[0]
    b1.arm_vehicle()
    b1.set_mode("GUIDED")
    b1.set_speed(b1.max_speed)

    counter = 1
    enemy_last_xy = None               # 上一次規劃時的敵船 xy
    while not stop_flag:
        # 全部船已經在等待圈，才會進入這段；你的判斷還在別處做就好
        if not first_plot_done:
            print("\n=== 所有船隻已抵達等待圈，開始計算包圍路徑 ===")

        # 1) 取得敵船當前 xy
        enemy_now_xy = global_to_planning(
            enemy_boat.lat, enemy_boat.lon, global_origin
        )

        # 2) 判斷是否需要重算
        need_replan = False
        if enemy_last_xy is None:            # 第一次一定重算
            need_replan = True
            counter = 1
        elif planning_done:
            a = enemy_now_xy[0] - enemy_last_xy[0]
            b = enemy_now_xy[1] - enemy_last_xy[1]
            if math.hypot(a, b) > 3:       # 位移超過 3 m
                need_replan = True
                counter = 1
        if counter % 8 == 0 and planning_done:
            need_replan = True
            counter = 1
        counter += 1

        # 3) 要重算就上鎖 → 呼叫 cal_surrounding_path()
        if need_replan:
            with planning_lock:
                cal_surrounding_path(enemy_boat)
                for b in boats:
                    if hasattr(b, "path_version"):
                        b.path_version += 1
            enemy_last_xy = enemy_now_xy     # 更新基準點

        time.sleep(1)   # 每 2 秒檢查一次

# =========================
# ★ Boat Brain Thread
# =========================

def boat_brain(boat: BoatController, enemy_boat):
    global planning_done

    boat.arm_vehicle()
    boat.set_mode("GUIDED")
    boat.set_speed((boat.max_speed + boat.min_speed) / 2)
    boat.path_version = 0

    # --------------------------mode1----------------------------------------

    while not stop_flag and not planning_done:
        if boat.boat_id == 1:
            boat.set_position_target(enemy_boat.lat,enemy_boat.lon)
        elif not stop_follow:
            all_follow_leader(boats, 3, 5)
        time.sleep(1)

    # --------------------------mode2--------------------------------------

    print(f"船隻 {boat.boat_id} 開始包圍任務")

    if boat.get_mode() != "GUIDED":
        boat.set_mode("GUIDED")     # 解除 HOLD
        boat.set_speed(boat.max_speed)

    last_version = 0
    while not stop_flag:
        if not hasattr(boat, "current_path") or not boat.current_path:
            time.sleep(0.2)
            continue
        if last_version < boat.path_version:
            last_version = boat.path_version
            boat.set_speed(boat.max_speed)
            boat.move_along_path(boats)
        time.sleep(0.1)
        


# =========================
# ★ 主要流程
# =========================

def main():
    global boats, boats_data, global_origin
    # five own boats
    conn_ports=[14551,14552,14553,14554,14555]
    boats=[BoatController(f"udp:127.0.0.1:{p}") for p in conn_ports]

    # ➜➜➜ 在這裡補上 min_speed / max_speed 兩個屬性
    for b in boats:
        b.min_speed = 0.6     # 你允許的最慢速度 (m/s)
        b.max_speed = 1.2     # 你允許的最快速度 (m/s)

    # enemy boat
    elat, elon = planning_to_global(ENEMY_OFFSET, 0, (boats[0].msg_monitor.current_lat, boats[0].msg_monitor.current_lon)) # 設置敵船在東方120m處
    e_radius = obstacle_r
    enemy_boat = Enemy_boat(elat , elon, e_radius)

    global_origin=(boats[0].msg_monitor.current_lat, boats[0].msg_monitor.current_lon)
    config.origin_point = global_origin
    objects = []
    for b in boats: objects.append(b)
    objects.append(enemy_boat)

    # 統一 thread 管理
    threads = []
    threads.append(threading.Thread(target=watcher_thread, args=(enemy_boat,), daemon=True))
    # boat brains(boat流程)
    for b in boats:
        threads.append(threading.Thread(target=boat_brain, args=(b, enemy_boat), daemon=True))
    threads.append(threading.Thread(target=telemetry_printer, args=(enemy_boat,), daemon=True))
    threads.append(threading.Thread(target=enemy_keep_moving, args=(enemy_boat,), daemon=True))
    threads.append(threading.Thread(target=bubble_list_update_thread, args=(boats,), daemon=True))
    # draw bubbles（pygame 視覺化）
    threads.append(threading.Thread(target=draw_bubbles_thread, args=(objects,),daemon=True))
    # 避碰 threads（每艘船都一條）
    config.leader_avoidance_on = True
    for b in boats:
         threads.append(threading.Thread(target=collide_avoidance_thread, args=(b, objects), daemon=True))

    # 啟動所有 thread
    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("CTRL‑C … stopping")
    finally:
        global stop_flag; stop_flag=True
        time.sleep(1)
        for b in boats: b.disarm()
        print("All done.")

if __name__=="__main__":
    main()



"""
sim_vehicle.py -v rover -I 0 -w --frame=rover-boat --console --map --custom-location=24.7871,120.9947,0,0 --sysid=1 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551 --add-param-file=./ardupilot/Tools/autotest/complete_boat.parm

sim_vehicle.py -v rover -I 1 -w --frame=rover-boat --console --map --custom-location=24.7872,120.9946,0,0 --sysid=2 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14552 --add-param-file=./ardupilot/Tools/autotest/complete_boat.parm

sim_vehicle.py -v rover -I 2 -w --frame=rover-boat --console --map --custom-location=24.7870,120.9946,0,0  --sysid=3 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14553 --add-param-file=./ardupilot/Tools/autotest/complete_boat.parm

sim_vehicle.py -v rover -I 3 -w --frame=rover-boat --console --map --custom-location=24.7873,120.9945,0,0  --sysid=4 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14554 --add-param-file=./ardupilot/Tools/autotest/complete_boat.parm

sim_vehicle.py -v rover -I 4 -w --frame=rover-boat --console --map --custom-location=24.7869,120.9945,0,0 --sysid=5 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14555 --add-param-file=./ardupilot/Tools/autotest/complete_boat.parm

sim_vehicle.py -v rover -I 10 -w --frame=rover-boat --console --map --custom-location=24.7871,120.9966,0,0 --sysid=10 --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14561 --add-param-file=./ardupilot/Tools/autotest/complete_boat.parm


"""