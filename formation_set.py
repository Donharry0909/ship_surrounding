# formation_follow.py
"""
可自訂「間格距離 spacing」的五種隊形跟隨
- spacing (float)：單位公尺，影響所有隊形
- 第五隊形（圓形）動態計算，1號船正上方，2~5按角度排列
"""

import math
import time
import config
from Missionplanner_utils import BoatController

SPACING = 12
OFFSET_R = 0.5

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

def gen_formations(spacing):
    """回傳 5 種隊形；最外層 key=formation_id, 內層 key=boat_id"""
    return {
        1: {  # 縱列
            1: (0, 0),
            2: (0, -spacing),
            3: (0, -2 * spacing),
            4: (0, -3 * spacing),
            5: (0, -4 * spacing),
        },
        2: {  # 橫列
            1: (0, 0),
            2: (-spacing, 0),
            3: (-2 * spacing, 0),
            4: (spacing, 0),
            5: (2 * spacing, 0),
        },
        3: {  # V 型
            1: (0, 0),
            2: (-spacing, -spacing),
            3: (-2 * spacing, -2 * spacing),
            4: (spacing, -spacing),
            5: (2 * spacing, -2 * spacing),
        },
        4: {  # X / 菱形
            1: (0, 0),
            2: (-spacing, spacing),
            3: (-spacing, -spacing),
            4: (spacing, spacing),
            5: (spacing, -spacing),
        },
        5: gen_circle_formation(spacing),  # 動態圓形
    }

def gen_circle_formation(spacing):
    """
    生成圓形隊形，spacing 為圓心到每艘船的距離，n為船數
    1號正上方(0度)，2左上，3左下，4右上，5右下，逆時針分佈
    """
    positions = [(0, 0)]  # 領頭在正中心
    radius = spacing      # spacing作為半徑

    # 指定五個角度，讓分佈有「1上，2左上，3左下，4右上，5右下」這種隊型
    angles_deg = [90, 150, 210, 30, 330]  # 逆時針：90度在正上

    for ang in angles_deg:
        rad = math.radians(ang)
        dx = radius * math.cos(rad)
        dy = radius * math.sin(rad)
        positions.append((dx, dy))
    return positions[:5]  # 剛好5艘

def _rotate(dx, dy, heading_deg):
    """依 heading (deg, 0=北、順時針+) 旋轉點"""
    rad = math.radians(heading_deg)
    xr = dx * math.cos(rad) - dy * math.sin(rad)
    yr = dx * math.sin(rad) + dy * math.cos(rad)
    return xr, yr

def gen_destinations(leader, boats, formation, offset_r):
    """
    依 leader 位置/航向，計算各船目標經緯度。
    leader 需具備 lat, lon, heading 屬性；boats 內每艘需有 boat_id。
    回傳 {boat_id: (lat, lon)}。
    """
    origin = config.origin_point  # 或用 (leader.lat, leader.lon)
    leader_x, leader_y = global_to_planning(leader.msg_monitor.current_lat, leader.msg_monitor.current_lon, origin)

    heading = leader.msg_monitor.heading
    offset_dx, offset_dy = _rotate(0, offset_r, -heading)
    destinations = {}
    for boat in boats:
        rel = formation[boat.boat_id]
        dx_r, dy_r = _rotate(rel[0], rel[1], -heading)
        tx = leader_x + dx_r + offset_dx
        ty = leader_y + dy_r + offset_dy
        tlat, tlon = planning_to_global(tx, ty, origin)
        destinations[boat.boat_id] = (tlat, tlon)
    return destinations

def all_follow_leader(boats, f_num, offset_r):
    """
    持續讓所有船依照隊形跟隨 leader。
    offset_r: 隊形前移距離
    """
    formation_list = gen_formations(SPACING)
    formation = formation_list[f_num]
    leader = boats[0]
    destinations = gen_destinations(leader, boats, formation, offset_r)
    for boat in boats:
        if boat.boat_id != 1:
            latlon = destinations[boat.boat_id]
            boat.set_position_target(*latlon)
            boat.set_speed(boat.max_speed)

