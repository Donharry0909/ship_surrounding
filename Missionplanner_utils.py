# Missionplanner_utils.py

import threading
from pymavlink import mavutil
import time
import math
import numpy as np
import config

class MessageMonitor(threading.Thread):
    """
    訊息監控類別，用於在獨立線程中監控和接收來自無人載具的訊息。
    """
    def __init__(self, vehicle):
        """
        初始化訊息監控器。
        
        Args:
            vehicle: mavutil.mavlink_connection 物件，表示與無人載具的連接。
        """
        super().__init__(daemon=True)
        # 先把 vehicle 存起來
        self.vehicle = vehicle
        
        # 初始化所有欄位
        self.wp_dist      = 0.0
        self.groundspeed  = None
        self.heading      = None
        self.current_lat  = None
        self.current_lon  = None
        self.boat_id      = None
        self.running      = True
        self.initialized  = False
        self._lock        = threading.Lock()
        
        # 等個心跳第一筆訊息，拿到 boat_id
        try:
            msg = self.vehicle.recv_match(type=['VFR_HUD','GLOBAL_POSITION_INT','NAV_CONTROLLER_OUTPUT'],
                                          blocking=True, timeout=5)
            if msg:
                self.boat_id = msg.get_srcSystem()
                print(f"[MsgMon] 船隻 ID: {self.boat_id} 已接收首筆訊息")
        except Exception as e:
            print(f"[MsgMon] 首筆訊息接收錯誤: {e}")

    def check_params(self):
        """
        檢查所有參數是否都有值。
        
        Returns:
            bool: 如果所有參數都有值，則返回 True，否則返回 False。
        """
        return (isinstance(self.wp_dist, (int, float)) and
                isinstance(self.groundspeed, (int, float)) and
                isinstance(self.heading, (int, float)) and
                isinstance(self.current_lat, (int, float)) and
                isinstance(self.current_lon, (int, float)) and
                isinstance(self.boat_id, (int, str)) and  # 檢查 boat_id
                self.boat_id is not None)  # 確保 boat_id 不為 None

    def run(self):
        """
        線程執行函數，持續監控狀態。
        """
        while self.running:
            self.monitor_status()  # 監控狀態
            time.sleep(0.1)  # 休眠 0.1 秒

    def monitor_status(self):
        """
        監控無人載具的狀態，接收並解析訊息。
        """
        try:
            msg = self.vehicle.recv_match(type=['VFR_HUD', 'GLOBAL_POSITION_INT','NAV_CONTROLLER_OUTPUT'], blocking=True)
            if msg:
                self.boat_id = msg.get_srcSystem()  # 獲取船隻 ID
                # print(f"船隻 {self.boat_id} 收到消息: {msg.get_srcSystem()}")  # 船隻ID
                if msg.get_type() == 'VFR_HUD':
                    # print(f"船隻 {self.boat_id} 速度: {msg.groundspeed} m/s, 航向: {msg.heading}度")
                    self.groundspeed = msg.groundspeed  # 獲取地面速度
                    self.heading = msg.heading  # 獲取航向
                elif msg.get_type() == 'GLOBAL_POSITION_INT':
                    # print(f"船隻 {self.boat_id} 位置: 緯度={msg.lat/1e7}, 經度={msg.lon/1e7}")
                    self.current_lat = msg.lat / 1e7  # 獲取緯度
                    self.current_lon = msg.lon / 1e7  # 獲取經度
                elif msg.get_type() == 'NAV_CONTROLLER_OUTPUT':
                    self.wp_dist = msg.wp_dist  # 獲取距離下一個航點的距離
                    # print(f"船隻 {self.boat_id} 距離下一個路徑點: {msg.wp_dist}米")
                
                # 檢查所有參數
                if not self.initialized and self.check_params():
                    self.initialized = True  # 標記為已初始化
        except Exception as e:
            print(f"船隻 {self.boat_id} 消息接收錯誤: {str(e)}")

    def update_status(self):
        """
        手動更新狀態。
        """
        with self._lock:
            # print(f"手動更新船隻 {self.boat_id} 狀態...")
            self.monitor_status()  # 監控狀態

    def is_initialized(self):
        """
        檢查是否初始化完成。
        
        Returns:
            bool: 如果已初始化，則返回 True，否則返回 False。
        """
        with self._lock:
            return self.initialized

    def stop(self):
        """
        停止訊息監控器。
        """
        self.running = False  # 停止執行
        

class BoatController:
    """
    船隻控制器類別，用於控制無人船的移動和狀態。
    """
    def __init__(self, connection_string):
        """
        初始化船隻控制器。
        
        Args:
            connection_string: 連接字串，用於連接到無人船。
        """
        self.vehicle = mavutil.mavlink_connection(connection_string)  # 建立與無人船的連接
        self.request_message_interval(0, 1)  # 設定接收訊息間隔
        # self.vehicle.wait_heartbeat()
        print(f"Vehicle connected on {connection_string}!")
        
        self.target_system = self.vehicle.target_system  # 目標系統 ID
        self.target_component = self.vehicle.target_component  # 目標組件 ID
        self.msg_monitor = MessageMonitor(self.vehicle)  # 建立訊息監控器
        self.msg_monitor.start()  # 啟動訊息監控器
        while not self.msg_monitor.is_initialized():
            self.msg_monitor.update_status()  # 手動更新狀態，直到初始化完成
        self.boat_id = self.msg_monitor.boat_id  # 船隻 ID
        self.current_lat = self.msg_monitor.current_lat  # 當前緯度
        self.current_lon = self.msg_monitor.current_lon  # 當前經度
        self.groundspeed = self.msg_monitor.groundspeed  # 地面速度
        self.heading = self.msg_monitor.heading  # 航向
        self.arrival_threshold = 3.0  # 到達目標點的距離閾值
        self.target_speed = 1.2  # 目標速度
        self.current_path = []
        self.min_speed = 0.6  # 最小速度
        self.max_speed = 1.2  # 最大速度
        self.avoidance_flag = False

    def set_mode(self, mode_name):
        """
        設定飛行模式。
        
        Args:
            mode_name: 模式名稱，例如 "GUIDED"。
        
        Returns:
            bool: 如果模式切換成功，則返回 True，否則返回 False。
        """
        try:
            mode_id = self.vehicle.mode_mapping()[mode_name]  # 獲取模式 ID
            print(f"船隻 {self.boat_id} 切換至 {mode_name} 模式 (ID: {mode_id})")
            
            self.vehicle.mav.set_mode_send(
                self.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id)
            
            start_time = time.time()
            while time.time() - start_time < 5:
                msg = self.vehicle.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                if msg and msg.custom_mode == mode_id:
                    print(f"船隻 {self.boat_id} 已成功切換至 {mode_name} 模式")
                    return True
                time.sleep(0.1)
            return False
        except Exception as e:
            print(f"模式切換錯誤: {str(e)}")
            return False
            
    def set_speed(self, speed):
        """
        設定船隻速度。
        
        Args:
            speed: 目標速度 (m/s)。
        """
        self.target_speed = speed
        
        self.vehicle.mav.command_long_send(
        self.target_system,
        self.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,  # 確認標誌
        0,  # 速度類型 0 = 地面速度
        self.target_speed,  # 速度值(m/s)
        -1,  # 油門百分比(-1表示不變)
        0, 0, 0, 0
        )

    def request_message_interval(self, message_id: int, frequency_hz: float):
        """
        請求 MAVLink 訊息的頻率。
        
        Args:
            message_id (int): MAVLink 訊息 ID。
            frequency_hz (float): 頻率 (Hz)。
        """
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            message_id, # The MAVLink message ID
            1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
            0, 0, 0, 0, # Unused parameters
            0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
        )

    def arm_vehicle(self):
        """
        解鎖船隻。
        
        Returns:
            bool: 如果解鎖成功，則返回 True，否則返回 False。
        """
        print(f"船隻 {self.boat_id} 正在解鎖船隻...")
        self.vehicle.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0)
        
        return self.wait_for_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

    def disarm(self):
        """
        上鎖船隻。
        
        Returns:
            bool: 如果上鎖成功，則返回 True，否則返回 False。
        """
        print("Sending disarm command...")
        self.vehicle.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0)
        return self.wait_for_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

    def wait_for_command_ack(self, command_id, timeout=3):
        """
        等待命令確認。
        
        Args:
            command_id: 命令 ID。
            timeout: 超時時間 (秒)。
        
        Returns:
            bool: 如果收到確認，則返回 True，否則返回 False。
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
            if msg and msg.command == command_id:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"Command {command_id} accepted")
                    return True
                else:
                    print(f"Command {command_id} failed with result: {msg.result}")
                    return False
        print(f"Command {command_id} timed out")
        return False
    
    def hold_position(self):
        """清掉目標 → 送 0 速 → 切到 HOLD。"""
        self.stop_vehicle()      # 把 active position‑target + 速度歸零
        self.set_mode("HOLD")    # Rover 的 HOLD 會立刻停車 :contentReference[oaicite:1]{index=1}

    def stop_vehicle(self):
        """
        停止船隻移動。
        """
        # 使用 mavlink 訊息獲取當前航向
        while True:
            msg = self.vehicle.recv_match(type='ATTITUDE', blocking=True, timeout=1)
            if msg:
                current_yaw = msg.yaw
                break
            else:
                current_yaw = 0  # 若無法獲取則設為0
        
        # 設定 type_mask，保持航向但停止所有移動
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        
        # 發送停止命令並保持當前航向
        self.vehicle.mav.set_position_target_global_int_send(
            0,                      
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            type_mask,
            0, 0, 0,               # 位置 (不使用)
            0, 0, 0,               # 速度設為零
            0, 0, 0,               # 加速度設為零
            current_yaw, 0         # 保持當前航向，航向角速度為零
        )
        time.sleep(0.1)  # 確保命令被執行

    def set_position_target(self, lat, lon, alt=0):
        """
        設定目標位置。
        
        Args:
            lat: 目標緯度。
            lon: 目標經度。
            alt: 目標高度 (預設為 0)。
        
        Returns:
            bool: 如果設定成功，則返回 True，否則返回 False。
        """
        try:
            lat_int = int(lat * 1e7)  # 將緯度轉換為整數
            lon_int = int(lon * 1e7)  # 將經度轉換為整數
            alt_int = int(alt * 1000)  # 將高度轉換為整數
            
            # 設置類型遮罩，只使用位置控制
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )
            
            self.vehicle.mav.set_position_target_global_int_send(
                0,                      
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
                type_mask,
                lat_int, lon_int, alt_int,
                0, 0, 0,  # 速度
                0, 0, 0,  # 加速度
                0, 0      # 航向和航向角速度
            )
            
            self.last_target = (lat, lon)
            return True
                
        except Exception as e:
            print(f"船隻 {self.boat_id} 設置位置目標時出錯: {str(e)}")
            return False
        
    def get_mode(self):
        """
        獲取當前飛行模式。
        
        Returns:
            str: 模式名稱，如果無法獲取則返回 None。
        """
        msg = self.vehicle.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg:
            mode_id = msg.custom_mode
            for mode_name, mode_number in self.vehicle.mode_mapping().items():
                if mode_number == mode_id:
                    return mode_name
        return None
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        計算兩點間的實際距離（公尺）。
        
        Args:
            lat1: 起點緯度。
            lon1: 起點經度。
            lat2: 終點緯度。
            lon2: 終點經度。
        
        Returns:
            float: 兩點間的距離 (公尺)。
        """
        R = 6371000  # 地球半徑(公尺)
        lat1, lon1 = math.radians(lat1), math.radians(lon1)  # 轉換為弧度
        lat2, lon2 = math.radians(lat2), math.radians(lon2)  # 轉換為弧度
        
        dlat = lat2 - lat1  # 緯度差
        dlon = lon2 - lon1  # 經度差
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
            math.cos(lat1) * math.cos(lat2) * 
            math.sin(dlon/2) * math.sin(dlon/2))  # Haversine 公式的一部分
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))  # Haversine 公式的一部分
        
        return R * c  # 計算距離
    
        
    
    def move_to_point(self, target_lat, target_lon):   # 注意若avoidance_flag == true 速度以collide_control為準
        """
        移動到目標點。
        
        Args:
            target_lat: 目標緯度。
            target_lon: 目標經度。
        
        Returns:
            bool: 如果移動成功，則返回 True，否則返回 False。
        """
        while not config.stop_flag:  # 當 config.stop_flag 為 False 時執行
            # 設定初始速度
            if not self.avoidance_flag:
                self.set_speed(self.target_speed)  # 設定目標速度
            self.set_position_target(target_lat, target_lon)  # 設定目標位置
            time.sleep(0.5)  # 等待命令被執行

            try:
                interval = 0
                while not config.stop_flag:  # 當 config.stop_flag 為 False 時執行
                    self.msg_monitor.update_status()
                    # ★ 若避碰旗標亮，不改速度 / 位置
                    if self.avoidance_flag:
                        time.sleep(0.2)
                        continue
                    
                    self.set_speed(self.target_speed)
                    # 更新當前位置
                    current_distance = self.calculate_distance(
                        self.msg_monitor.current_lat,
                        self.msg_monitor.current_lon,
                        target_lat,
                        target_lon
                    )  # 計算當前位置與目標位置的距離

                    # 檢查是否到達
                    if current_distance < self.arrival_threshold:  # 如果距離小於閾值
                        self.avoidance_flag = False   # ★ 收掉避碰狀態
                        print(f"船隻 {self.boat_id} 已到達目的地")
                        self.stop_vehicle()  # 停止船隻移c動
                        break  # 跳出迴圈
                        
                    time.sleep(0.5)  # 休眠 0.5 秒
                    if interval % 80 == 0:
                        print(f"船隻 {self.boat_id} 距離目標點: {current_distance}米 船速：{self.target_speed}m/s")
                    interval += 1           
                return True  # 移動成功               
            except Exception as e:
                print(f"移動過程出錯: {type(e).__name__}: {e}")
                return False  # 移動失敗
            
    def move_along_path(self, boats):
        """沿著規劃的路徑移動"""
        if not self.current_path:
            print(f"船隻 {self.boat_id} 沒有規劃路徑，無法移動")
            return False
        
        local_version = self.path_version          # ★ 抓當前版本
        path_length = len(self.current_path)
        current_waypoint_index = 0

        # 等待所有船隻都完成路徑規劃
        all_ready = False
        while not all_ready and not config.stop_flag:
            all_ready = all([hasattr(boat, "total_distance") for boat in boats])
            if not all_ready:
                time.sleep(0.1)

        if config.stop_flag:
            return False

        # 緩加速相關參數
        current_speed = 0.2  # 起始速度
        target_speed = self.target_speed
        acceleration_rate = 0.05  # 降低加速率
        update_interval = 0.2  # 增加更新間隔
        min_speed = self.min_speed  # 最小速度
        max_speed = self.max_speed  # 最大速度
        speed_check_counter = 80  # 用於檢查速度的計數器

        # 計算所有船隻中最長的路徑距離
        # Python 3.8+ 有 default 參數，直接用最乾淨
        max_distance = max([boat.total_distance for boat in boats])
        if max_distance != 0:
            adjusted = self.total_distance / max_distance
            # 確保速度在允許範圍內
            adjusted_speed = max(adjusted * self.max_speed, self.min_speed)
            self.set_speed(adjusted_speed)
            print(f"船隻 {self.boat_id} 調整速度為 {adjusted_speed:.2f} m/s")

        # 發送第一個目標點
        current_waypoint = self.current_path[current_waypoint_index]
        self.set_position_target(current_waypoint[0], current_waypoint[1])
        print(
            f"船隻 {self.boat_id} 移動到路徑點 {current_waypoint_index + 1}/{path_length}"
        )

        while current_waypoint_index < path_length and not config.stop_flag:
            try:
                # ====== 版本變了就退出 ======
                if self.path_version != local_version:  # 發現外面換路徑
                    print(f"船隻 {self.boat_id} 偵測到新路徑，重新規劃")
                    return False                        # 讓外層再 call
                if self.avoidance_flag:
                    time.sleep(0.1)
                    continue
                if self.groundspeed < 0.1:
                    self.set_speed(min_speed)
                # 更新當前位置
                self.msg_monitor.update_status()
                self.current_lat = self.msg_monitor.current_lat
                self.current_lon = self.msg_monitor.current_lon
                self.groundspeed = self.msg_monitor.groundspeed

                # 計算到當前目標點的距離
                current_distance = self.calculate_distance(
                    self.current_lat,
                    self.current_lon,
                    current_waypoint[0],
                    current_waypoint[1],
                )

                # 緩加速控制
                if current_speed < target_speed and current_distance > 10:  # 距離大於10米時才加速
                    if abs(self.groundspeed - current_speed) < 0.2:
                        current_speed = min(
                            current_speed + acceleration_rate, target_speed
                        )
                        self.set_speed(current_speed)
                        print(f"船隻 {self.boat_id} 加速至: {current_speed:.2f} m/s")

                # 減速控制
                elif current_distance < 10:  # 距離小於10米開始減速
                    self.set_speed(1.0)

                # 每80次迴圈輸出一次狀態
                speed_check_counter += 1
                if speed_check_counter >= 80:
                    print(
                        f"船隻 {self.boat_id} 狀態 - "
                        f"目標速度: {current_speed:.2f} m/s, "
                        f"實際速度: {self.groundspeed:.2f} m/s, "
                        f"距離: {current_distance:.2f}m, "
                        f"目標點: {current_waypoint_index + 1}/{path_length}"
                    )
                    speed_check_counter = 0


                # 檢查是否到達目標點
                if current_distance <= self.arrival_threshold:
                    print(
                        f"船隻 {self.boat_id} 已到達路徑點 {current_waypoint_index + 1}"
                    )
                    current_waypoint_index += 1

                    # 如果還有下一個點，立即設定下一個目標
                    if current_waypoint_index < path_length:
                        current_waypoint = self.current_path[current_waypoint_index]
                        self.set_position_target(
                            current_waypoint[0], current_waypoint[1]
                        )
                        print(
                            f"船隻 {self.boat_id} 移動到路徑點 {current_waypoint_index + 1}/{path_length}"
                        )
                        if current_waypoint_index == path_length - 1:
                            self.set_speed(min_speed)
                    else:
                        # 到達最後一個點，切換到 HOLD 模式
                        print(f"船隻 {self.boat_id} 到達最終目標點")
                        self.stop_vehicle()
                        break

                time.sleep(0.1)

            except Exception as e:
                print(f"船隻 {self.boat_id} 移動過程出錯: {str(e)}")
                return False

        print(f"船隻 {self.boat_id} 完成路徑移動")
        return True
    
    def move_to_point_v2(self, target_lat, target_lon):
        """
        移動到目標點。
        
        Args:
            target_lat: 目標緯度。
            target_lon: 目標經度。
        
        Returns:
            bool: 如果移動成功，則返回 True，否則返回 False。
        """
        while not config.stop_flag:  # 當 stop_flag 為 False 時執行
            # 設定初始速度
            # self.set_speed(1.2)  # 設定目標速度
            self.set_position_target(target_lat, target_lon)  # 設定目標位置
            self.set_speed(self.target_speed)  # 設定目標速度
            
            time.sleep(1)  # 等待命令被執行

            try:
                while not config.stop_flag:  # 當 stop_flag 為 False 時執行
                    # 更新當前位置
                    current_distance = self.calculate_distance(
                        self.msg_monitor.current_lat,
                        self.msg_monitor.current_lon,
                        target_lat,
                        target_lon
                    )  # 計算當前位置與目標位置的距離
                    print(f"船隻 {self.boat_id} 距離目標點: {current_distance}米 船速：{self.target_speed}m/s")

                    # 檢查是否到達
                    if current_distance < self.arrival_threshold:  # 如果距離小於閾值
                        print(f"已到達目的地")
                        self.stop_vehicle()  # 停止船隻移動
                        break  # 跳出迴圈
                        
                    time.sleep(0.5)  # 休眠 0.5 秒
                        
                return True  # 移動成功
                
            except Exception as e:
                print(f"移動過程出錯: {str(e)}")
                return False  # 移動失敗
    
    def move_along_path_single(self, path):
        """
        依照預先定義的路徑 (一系列緯度、經度) 移動船隻。

        Args:
            path (list of tuples): [(lat1, lon1), (lat2, lon2), ...] 
                                代表船隻應該依序移動的路徑點。

        Returns:
            bool: 成功完成路徑則回傳 True，若中途發生錯誤則回傳 False。
        """
        try:
            for idx, (lat, lon) in enumerate(path):
                print(f"船隻 {self.boat_id} 正在前往第 {idx+1} 個航點: 緯度={lat}, 經度={lon}")
                
                success = self.move_to_point_v2(lat, lon)
                if not success:
                    print(f"船隻 {self.boat_id} 在航點 {idx+1} 發生錯誤，中斷導航")
                    return False  # 若移動失敗則中斷任務
                
                print(f"船隻 {self.boat_id} 成功到達航點 {idx+1}")
            
            print(f"船隻 {self.boat_id} 已完成整個航線")
            return True
        
        except Exception as e:
            print(f"船隻 {self.boat_id} 在 move_along_path 發生錯誤: {str(e)}")
            return False
        
    def reached_target(self, tol=1.0):
        """
        判斷是否已抵達最近一次 set_position_target 的目標點
        tol: 允許誤差 (公尺)
        """
        if getattr(self, "last_target", None) is None:
            return False

        # ① 直接算地理距離
        d = self.calculate_distance(
            self.msg_monitor.current_lat,
            self.msg_monitor.current_lon,
            self.last_target[0],
            self.last_target[1]
        )
        return d <= tol


    def calculate_surround_positions(
        self, center_lat, center_lon, num_boats=5, radius=15
    ):
        """
        計算圍繞中心點的GPS位置

        Args:
            center_lat: 中心點緯度
            center_lon: 中心點經度
            num_boats: 船隻數量
            radius: 包圍半徑(公尺)

        Returns:
            list: 包含每艘船目標位置的列表 [(lat1,lon1), (lat2,lon2), ...]
        """
        positions = []

        # 將半徑從公尺轉換為經緯度差
        # 緯度: 1度 ≈ 111公里
        lat_offset = radius / 111000.0
        # 經度: 1度 = cos(緯度) * 111公里
        lon_offset = radius / (111000.0 * np.cos(np.radians(center_lat)))

        # 計算每艘船的位置
        for i in range(num_boats):
            # 計算圓周上的角度 (平均分配)
            angle = (2 * np.pi * i) / num_boats

            # 計算偏移量
            d_lat = lat_offset * np.cos(angle)
            d_lon = lon_offset * np.sin(angle)

            # 計算最終位置
            target_lat = center_lat + d_lat
            target_lon = center_lon + d_lon

            positions.append((target_lat, target_lon))
            print(f"船隻 {i} 目標位置: 緯度={target_lat:.7f}, 經度={target_lon:.7f}")

        return positions
    