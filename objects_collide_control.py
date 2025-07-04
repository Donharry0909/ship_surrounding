# objects_collide_control.py
import math, time
import config
from math import inf
from Missionplanner_utils import BoatController
from typing import Tuple
import pygame
import copy

SEMI_MAJOR = 5
SEMI_MINOR = 2.5
AVOID_DISTANCE = 10
SAVE_DISTANCE = 3
SCAN_HALF_DEG   = 3      # 前向左右各掃描 3°

global bubbles_list, start_avoidance_flag
start_avoidance_flag = False
bubbles_list = {}

SCREEN_W, SCREEN_H = 800, 800
SCALE = 8              # 1 m → 2 pixels
Piority_list = [1, 2, 3, 4 ,5]

RADIUS_M = SEMI_MAJOR * 2.5
SPAN_DEG = 30

# ----------------------------- 座標轉換公式 -----------------------------------------

def xy_to_screen(x_m, y_m):
    sx = SCREEN_W  // 2 + int(x_m * SCALE)
    sy = SCREEN_H // 2 - int(y_m * SCALE)   # y 方向反過來
    return sx, sy

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

class Boat_avoidance_bubble():
    def __init__(self, boat_id, boat_lat, boat_lon, angle_rad, major, minor):
        boat_position = global_to_planning(boat_lat, boat_lon, config.origin_point)
        self.id = boat_id
        # 存原始大小
        self.base_major = major
        self.base_minor = minor
        # 当前使用的大小
        self.semi_major = major
        self.semi_minor = minor

        # 焦点、角度照旧
        self.focus = boat_position
        self.angle_rad = angle_rad
        self._recalc_geometry()

    def _recalc_geometry(self):
        # 计算焦距 c = sqrt(a^2 - b^2)
        c = math.sqrt(self.semi_major**2 - self.semi_minor**2)
        # 后焦点→中心向量
        dx_fc = c * math.cos(self.angle_rad)
        dy_fc = c * math.sin(self.angle_rad)
        self.center = (self.focus[0] + dx_fc,
                       self.focus[1] + dy_fc)
        # 中心→边界向量（a）
        dx_ce = self.semi_major * math.cos(self.angle_rad)
        dy_ce = self.semi_major * math.sin(self.angle_rad)
        self.edge_point = (self.center[0] + dx_ce,
                           self.center[1] + dy_ce)

    def update_info(self, boat_lat, boat_lon, angle_rad):
        boat_position = global_to_planning(boat_lat, boat_lon, config.origin_point)
        self.focus = boat_position
        self.angle_rad = angle_rad
        self._recalc_geometry()

    def adjust_size(self, current_speed, max_speed):
        """
        根据速度比例调整椭圆大小：
        ratio = current_speed / max_speed ∈ [0,1]
        """
        ratio = max(0.3, min(1.0, current_speed / max_speed))
        self.semi_major = self.base_major * ratio
        self.semi_minor = self.base_minor * ratio
        self._recalc_geometry()

    def get_focus_to_center(self):
        return (self.center[0] - self.focus[0],
                self.center[1] - self.focus[1])

    def get_center_to_edge(self):
        return (self.edge_point[0] - self.center[0],
                self.edge_point[1] - self.center[1])
    

# ---------- 共用工具 ----------

def haversine(lat1, lon1, lat2, lon2) -> float:
    """兩 GPS 點直線距離 (m)"""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def distance_cal(p1, p2):
    """p1/p2 = (lat, lon)"""
    return haversine(p1[0], p1[1], p2[0], p2[1])

def center_distance(b1: Boat_avoidance_bubble, b2: Boat_avoidance_bubble) -> float:
    x1, y1 = b1.center
    x2, y2 = b2.center
    return math.hypot(x1 - x2, y1 - y2)


def _inside_ellipse(px: float, py: float,
                    center: Tuple[float, float],
                    a: float, b: float, theta: float,
                    tol: float = 1e-8) -> bool:
    """
    檢查一點 (px,py) 是否落在旋轉橢圓內部／邊界
    橢圓：中心 center，長短軸 a,b，逆時針旋轉 theta
    """
    # 先平移，再反旋轉
    dx, dy = px - center[0], py - center[1]
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)
    x = cos_t*dx - sin_t*dy
    y = sin_t*dx + cos_t*dy
    return (x*x)/(a*a) + (y*y)/(b*b) <= 1.0 + tol

def bubbles_intersect(b1, b2,
                      num_samples: int = 36,
                      tol: float = 1e-8) -> bool:
    """
    粗略判斷兩顆 Boat_avoidance_bubble 是否相交
    先用外接圓快篩，再用邊界採樣精篩
    """
    # ---------- 0. 外接圓快篩 ----------
    dx = b1.center[0] - b2.center[0]
    dy = b1.center[1] - b2.center[1]
    dist_sq = dx*dx + dy*dy
    r1 = max(b1.semi_major, b1.semi_minor)
    r2 = max(b2.semi_major, b2.semi_minor)
    if dist_sq > (r1 + r2)**2:
        return False

    # ---------- 1. 決定採樣對象 ----------
    # 採樣較小橢圓，減少計算量
    samp, tgt = (b1, b2) if r1 < r2 else (b2, b1)

    # ---------- 2. 邊界採樣 ----------
    cos_s, sin_s = math.cos(samp.angle_rad), math.sin(samp.angle_rad)
    for k in range(num_samples):
        ang = 2.0 * math.pi * k / num_samples      # 在橢圓局部座標的角度
        # 橢圓標準參數方程
        ex = samp.semi_major * math.cos(ang)
        ey = samp.semi_minor * math.sin(ang)
        # 旋轉 + 平移到全域座標
        px = samp.center[0] + cos_s*ex - sin_s*ey
        py = samp.center[1] + sin_s*ex + cos_s*ey

        if _inside_ellipse(px, py,
                           tgt.center,
                           tgt.semi_major, tgt.semi_minor,
                           tgt.angle_rad, tol):
            return True

    return False

def anypoint_in_sector(bubbleA, bubbleB, num_samples: int = 36) -> bool:
    """
    判斷 bubbleB 的橢圓邊界是否「有任何一點」
    落在 bubbleA 前方 ±span_deg/2、且距離 ≤ radius_m 的扇形內。
    """
    radius_m = RADIUS_M
    span_deg = SPAN_DEG

    # ---------- 0. 外接圓快篩 ----------
    dx_f = bubbleB.center[0] - bubbleA.focus[0]
    dy_f = bubbleB.center[1] - bubbleA.focus[1]
    rB   = bubbleB.semi_major
    if math.hypot(dx_f, dy_f) - rB > radius_m:
        return False   # 整顆都離太遠，直接 fail

    # ---------- 1. 定一些常量 ----------
    half_span = math.radians(span_deg) / 2
    cosB, sinB = math.cos(bubbleB.angle_rad), math.sin(bubbleB.angle_rad)

    # ---------- 2. 在 bubbleB 邊界取樣 ----------
    for k in range(num_samples):
        t = 2.0 * math.pi * k / num_samples        # 0~2π
        # 橢圓局部座標 (ex, ey)
        ex = bubbleB.semi_major * math.cos(t)
        ey = bubbleB.semi_minor * math.sin(t)
        # 旋轉 + 平移到 global
        px = bubbleB.center[0] + cosB*ex - sinB*ey
        py = bubbleB.center[1] + sinB*ex + cosB*ey

        # ---------- 2a. 距離檢查 ----------
        dx = px - bubbleA.focus[0]
        dy = py - bubbleA.focus[1]
        dist = math.hypot(dx, dy)
        if dist > radius_m:
            continue   # 太遠

        # ---------- 2b. 角度檢查 ----------
        ang = math.atan2(dy, dx)
        diff = (ang - bubbleA.angle_rad + math.pi) % (2*math.pi) - math.pi
        if abs(diff) <= half_span:
            return True   # 命中！

    return False

def focus_in_sector(bubbleA, bubbleB) -> bool:
    """
    判斷 bubbleB 的焦點（即船 B 的實際位置）是否落在
    bubbleA 焦點為圓心、航向為中軸、±span_deg/2 的扇形內，且距離 <= radius_m。

    Parameters
    ----------
    bubbleA : 自己船的泡泡（提供圓心 & 航向）。
    bubbleB : 目標船的泡泡（取其 focus 作檢查點）。

    Returns
    -------
    True  → 船 B 在扇形內  
    False → 不在
    """
    span_deg = SPAN_DEG
    radius_m = RADIUS_M

    # ----- 1. 取兩船在 local-xy 座標的位置 -----
    ax, ay = bubbleA.focus           # 船 A 的位置 (m, m)
    bx, by = bubbleB.focus           # 船 B 的位置

    # ----- 2. 距離檢查 -----
    dx, dy = bx - ax, by - ay
    dist = math.hypot(dx, dy)
    if dist > radius_m:
        return False                 # 已經超出半徑，不用算角度

    # ----- 3. 角度檢查 -----
    dir_to_B = math.atan2(dy, dx)    # A → B 的方向
    # 角度差轉到 [-π, π)
    ang_diff = (dir_to_B - bubbleA.angle_rad + math.pi) % (2*math.pi) - math.pi
    half_span = math.radians(span_deg) / 2

    return abs(ang_diff) <= half_span

def a_pior_than_b(bubbleA: Boat_avoidance_bubble,
                  bubbleB: Boat_avoidance_bubble,
                  p_list) -> bool:
    """
    比較 2 艘船泡泡的優先級  
    回傳 True  →  bubbleA 的優先級高於 bubbleB  
    回傳 False →  bubbleA 優先級沒比較高（相同或更低）

    p_list 早排 > 晚排；若某船 ID 不在 p_list，視為最低。
    """
    try:
        idx_a = p_list.index(bubbleA.id)
    except ValueError:
        idx_a = len(p_list)          # 不在表內 → 最低

    try:
        idx_b = p_list.index(bubbleB.id)
    except ValueError:
        idx_b = len(p_list)

    return idx_a < idx_b             # 越前面 idx 越小 → 優先

def draw_avoidance_sector(screen, bubble: Boat_avoidance_bubble,
                          radius_m: float, span_deg: float = 180):
    """以 bubble.focus 為圓心、bubble.angle_rad 為中軸畫扇形"""
    # 轉成 pixel
    Rpx = int(radius_m * SCALE)
    cx, cy = xy_to_screen(*bubble.focus)

    # 取扇形頂點集合
    half   = math.radians(span_deg) / 2
    ang0   = bubble.angle_rad - half
    ang1   = bubble.angle_rad + half

    pts = [(cx, cy)]  # 圓心先放進去
    step = 6  # 每 6° 取一點，畫面夠順
    k = int(span_deg / step) + 1
    for i in range(k + 1):
        a = ang0 + i * (ang1 - ang0) / k
        px = cx + Rpx * math.cos(a)
        py = cy - Rpx * math.sin(a)   # Pygame y 軸向下
        pts.append((px, py))

    # 半透明面 + 外框
    surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(surf, (255, 80, 80, 40), pts)   # 內部淡紅
    pygame.draw.polygon(surf, (200,  0,  0, 180), pts, 1)
    screen.blit(surf, (0, 0))

def draw_rotated_ellipse(screen, bubble, color, width=2):
    """
    將 Boat_avoidance_bubble 依照 bubble.angle_rad 畫成旋轉橢圓
    （假設 bubble.focus → bubble.center → bubble.edge_point 共線且為長軸）
    """
    # --- 1. 把長、短軸長度換成 pixel ---
    a_px = int(bubble.semi_major * SCALE)   # 長軸 a
    b_px = int(bubble.semi_minor * SCALE)   # 短軸 b

    # --- 2. 在透明 surface 上先畫「水平橢圓」 ---
    surf = pygame.Surface((2 * a_px, 2 * b_px), pygame.SRCALPHA)
    pygame.draw.ellipse(
        surf,
        color,
        (0, 0, 2 * a_px, 2 * b_px),
        width,
    )

    # --- 3. 旋轉：注意 Pygame 角度正方向為逆時針、且 y 向下 ---
    deg = math.degrees(bubble.angle_rad)     # 把你的角度 (數學座標制, x+→右, y+→上) 轉成 Pygame
    rot = pygame.transform.rotate(surf, deg)

    # --- 4. 貼到主畫布，以 bubble.center 為中心 ---
    cx_px, cy_px = xy_to_screen(*bubble.center)
    rect = rot.get_rect(center=(cx_px, cy_px))
    screen.blit(rot, rect)

    # --- 5. 在橢圓「中心」畫 id（黑色字體，字小一點）---
    font = pygame.font.SysFont(None, 20)   # 字號 20，比預設小一點
    id_str = str(bubble.id)
    id_surf = font.render(id_str, True, (0, 0, 0))  # 黑色
    id_rect = id_surf.get_rect(center=(cx_px, cy_px))
    screen.blit(id_surf, id_rect)

    # 把焦點畫成小紅點、edge_point畫成綠點
    fx_px, fy_px = xy_to_screen(*bubble.focus)
    pygame.draw.circle(screen, (200, 80, 80), (fx_px, fy_px), 1)

    ex_px, ey_px = xy_to_screen(*bubble.edge_point)
    pygame.draw.circle(screen, (80, 200, 80), (ex_px, ey_px), 4)

def draw_others(screen, objs):
    for o in objs:
        # 不是我方船，而且帶有 radius
        if hasattr(o, "radius") and hasattr(o, "lat") and hasattr(o, "lon") and not isinstance(o, BoatController):
            x_m, y_m = global_to_planning(o.lat, o.lon, config.origin_point)
            sx, sy   = xy_to_screen(x_m, y_m)
            r_px     = int(o.radius * SCALE)

            temp = pygame.Surface((2*r_px, 2*r_px), pygame.SRCALPHA)
            pygame.draw.circle(temp, (255, 0, 0, 10), (r_px, r_px), r_px)     # 半透明
            pygame.draw.circle(temp, (255, 80, 80), (r_px, r_px), r_px, 2)    # 外框
            screen.blit(temp, (sx - r_px, sy - r_px))

def draw_path(screen, path, bubble_id, color=(0, 160, 255)):
    """
    把路徑 list[(lat,lon)…] 畫成藍點，最後一點畫叉叉
    """
    if not path:
        return
    path_color=(0, 160, 255)
    num_color = (0, 90, 180)
    last_idx = len(path) - 1
    for idx, (lat, lon) in enumerate(path):
        x_m, y_m = global_to_planning(lat, lon, config.origin_point)
        sx, sy   = xy_to_screen(x_m, y_m)

        if idx == last_idx:
            # 叉叉：兩條斜線
            size = 4
            pygame.draw.line(screen, path_color, (sx-size, sy-size), (sx+size, sy+size), 2)
            pygame.draw.line(screen, path_color, (sx-size, sy+size), (sx+size, sy-size), 2)

            # --- 小字 id，緊貼叉叉右上方 ---
            font = pygame.font.SysFont(None, 15)  # 比 bubble id 再小
            id_str = str(bubble_id)
            id_surf = font.render(id_str, True, num_color)
            id_rect = id_surf.get_rect(midleft=(sx + size + 2, sy - size - 2))
            screen.blit(id_surf, id_rect)

        else:
            # 中繼點：小圓
            pygame.draw.circle(screen, color, (sx, sy), 3)


def draw_bubbles_thread(objects):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    counter = 0

    while True:
        if start_avoidance_flag == True:
            break
        time.sleep(0.1)
    print(f"啟動pygame畫布")

    while not config.stop_flag:
        # --- 事件處理：允許關閉視窗 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                config.stop_flag = True

        screen.fill((255, 255, 255))   # 背景

        # --- 畫所有泡泡 ---
        # 這裡用 .copy() 避免邊讀邊寫衝突
        for bid, bubble in list(bubbles_list.items()):
            if bubble is None:
                continue
            draw_rotated_ellipse(screen, bubble, (0, 180, 200), width=2)

            # id／其他輔助文字
            fx_px, fy_px = xy_to_screen(*bubble.focus)
            txt = pygame.font.SysFont(None, 18).render(str(bid), True, (255, 255, 255))
            screen.blit(txt, (fx_px + 6, fy_px - 6))

        for obj in objects:
            if isinstance(obj, BoatController):
                # --- 6. 避障時用半透明紅色畫出偵測範圍 ---
                b = bubbles_list.get(obj.boat_id)           # 取對應的 Bubble
                if obj.avoidance_flag:   # ← 正在避障
                    draw_avoidance_sector(
                        screen,
                        b,
                        radius_m = RADIUS_M,  # 半徑 = 3×原長軸
                        span_deg = SPAN_DEG                    # 180° 半圓
                    )
                if hasattr(obj, "current_path") and obj.current_path:
                    draw_path(screen, obj.current_path, obj.boat_id)
                elif hasattr(obj, "last_target") and obj.last_target != None:
                    draw_path(screen, [obj.last_target], obj.boat_id)
                
        draw_others(screen, objects)


        # 畫面跟隨：若 bubble 超出畫面  則修改config.original_point 讓畫面跟隨bubble位置
        if not config.fixed_window:
            follow_bubble = bubbles_list[1]  # 你可自訂想跟隨的 id
            fx, fy = follow_bubble.focus
            # focus 離畫面中心的距離（就是 (0,0)）
            dist_from_center = math.hypot(fx, fy)
            if dist_from_center > 20 and counter % 10 == 0:   # 距離太遠就改畫面原點
                # 將 local-xy (fx, fy) 反推回新的 origin_point 經緯度
                config.origin_point = planning_to_global(fx, fy, config.origin_point)
                print(f"自動調整畫面中心到泡泡 ID {1}，新的 origin_point: {config.origin_point}")
            counter += 1

        pygame.display.flip()
        clock.tick(20)     # 20 FPS

    pygame.quit()

def bubble_list_update_thread(boats):
    global bubbles_list, start_avoidance_flag
    # 初始化
    print(f"啟動泡泡陣列更新函式")
    bubbles_list = {boat.boat_id : None for boat in boats}
    for boat in boats:
        boat.msg_monitor.update_status()
        bubble = Boat_avoidance_bubble(
            boat.boat_id,
            boat.msg_monitor.current_lat,
            boat.msg_monitor.current_lon,
            math.radians((90 - (boat.msg_monitor.heading or 0)) % 360),
            SEMI_MAJOR, SEMI_MINOR
        )
        print(f"船隻 {boat.boat_id} 建立防撞泡泡：\n"
              f"焦點(船隻)位置: {bubble.focus}  橢圓中心位置: {bubble.center}\n"
              f"角度：{(90 - (boat.msg_monitor.heading or 0)) % 360}\n"
              f"長軸: {bubble.semi_major} 短軸: {bubble.semi_minor}\n"
              f"---------------------------對照--------------------------------\n"
              f"實際角度(deg)：{boat.msg_monitor.heading}\n\n\n"
            )
        bubbles_list[boat.boat_id] = bubble

    start_avoidance_flag = True

    last_sp_list = {boat.boat_id : None for boat in boats}
    while True:  #更新
        for boat in boats:
            boat.msg_monitor.update_status()
            if  last_sp_list[boat.boat_id] != boat.msg_monitor.groundspeed or last_sp_list[boat.boat_id] is None:
                bubbles_list[boat.boat_id].adjust_size(
                    boat.msg_monitor.groundspeed,
                    boat.max_speed,
                )
                last_sp_list[boat.boat_id] = boat.msg_monitor.groundspeed
            bubbles_list[boat.boat_id].update_info(
                boat.msg_monitor.current_lat,
                boat.msg_monitor.current_lon,
                math.radians((90 - (boat.msg_monitor.heading or 0)) % 360),
            )
        time.sleep(0.001)

def collide_avoidance_thread(me, objects):
    p_list = Piority_list
    while True:
        if start_avoidance_flag == True:
            break
        time.sleep(0.1)
    print(f"船隻{me.boat_id}啟動避碰函式")
    my_bubble = bubbles_list[me.boat_id]
    last_speed = me.target_speed
    counter = 1

    while not config.stop_flag:
        # --- 掃描前方 ------------------------------------------------------
        if me.avoidance_flag == False:       # 若船隻不在避障模式跑這段
            for obj in objects:
                if isinstance(obj, BoatController):
                    if obj.boat_id == me.boat_id:
                        continue
                    tgt = obj
                    # 計算自身前方範圍是否進到target_boat船隻範圍
                    tgt_bubble = bubbles_list[tgt.boat_id]
                    if_intersect = bubbles_intersect(my_bubble, tgt_bubble)
                    if if_intersect and not tgt.avoidance_flag and (me.boat_id != 1 or config.leader_avoidance_on):
                        me_in_tgt_sector = anypoint_in_sector(tgt_bubble, my_bubble)  # 對方去掃我，發現我在範圍內
                        tgt_in_my_sector = anypoint_in_sector(my_bubble, tgt_bubble)  # 我去掃對方，發現對方在範圍內
                        me_pior_than_tgt = a_pior_than_b(my_bubble, tgt_bubble, p_list) # 我優先級高於對方
                        if tgt_in_my_sector:
                            if not (me_in_tgt_sector and me_pior_than_tgt):
                                print(f"***** 船隻{me.boat_id} 開始避障，障礙物為：{tgt.boat_id} *****")
                                last_speed = me.target_speed
                                me.avoidance_flag = True      # 確認附近有障礙物且優先級大於自己，開始避障
                                me.set_speed(0.1)
                        elif not me_in_tgt_sector:
                            if not me_pior_than_tgt:
                                print(f"***** 船隻{me.boat_id} 開始避障，障礙物為：{tgt.boat_id} *****")
                                last_speed = me.target_speed
                                me.avoidance_flag = True      # 確認附近有障礙物且優先級大於自己，開始避障
                                me.set_speed(0.1)
                else:
                    continue
                    # 暫時不考慮做
        else:
            continue_avoidance = False
            counter += 1
            if counter % 60 == 0:
                for obj in objects:
                    if isinstance(obj, BoatController):
                        if obj.boat_id == me.boat_id:
                            continue
                        tgt_bubble = bubbles_list[obj.boat_id]
                        copy_my_bubble = copy.deepcopy(my_bubble)
                        copy_my_bubble.semi_major = my_bubble.base_major
                        copy_my_bubble.semi_minor = my_bubble.base_minor
                        copy_my_bubble._recalc_geometry()
                        if bubbles_intersect(my_bubble, tgt_bubble):
                            if a_pior_than_b(my_bubble, tgt_bubble, p_list) is False:
                                continue_avoidance = True

                if continue_avoidance == False:
                    me.avoidance_flag = False
                    me.set_speed(last_speed)
                    print(f"***** 船隻{me.boat_id} 解除避障模式 預期速度：{last_speed}*****")

            if me.avoidance_flag:
                me.set_speed(0.1)
                                    

        time.sleep(0.05)
