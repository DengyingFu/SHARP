import numpy as np
from scipy.spatial.transform import Rotation as R
import math
# 通过调整欧拉角来修改一个四元数.
def adjust_quaternion_by_euler(quat_wxyz, yaw_adj_deg=0.0, pitch_adj_deg=0.0, roll_adj_deg=0.0):
    """
    通过调整欧拉角来修改一个四元数.

    参数:
    - quat_wxyz (np.array): 原始四元数, 格式为 [w, x, y, z].
    - yaw_adj_deg (float): 要增加的偏航角 (绕Z轴), 单位为度.
    - pitch_adj_deg (float): 要增加的俯仰角 (绕Y轴), 单位为度.
    - roll_adj_deg (float): 要增加的滚转角 (绕X轴), 单位为度.

    返回:
    - np.array: 调整后的新四元数, 格式为 [w, x, y, z].
    """
    print(f"原始四元数 (w, x, y, z): {quat_wxyz}")

    # --- 1. 格式转换: [w, x, y, z] -> [x, y, z, w] for scipy ---
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r_original = R.from_quat(quat_xyzw)

    # --- 2. 四元数 -> 欧拉角 ---
    # 使用 'zyx' 顺序, 对应 (偏航角, 俯仰角, 滚转角)
    euler_angles_rad = r_original.as_euler('zyx')
    print(f"原始欧拉角 (yaw, pitch, roll) in degrees: {np.rad2deg(euler_angles_rad)}")
    print("-" * 30)

    # --- 3. 修改欧拉角 ---
    print(f"计划调整角度 (yaw, pitch, roll): ({yaw_adj_deg}, {pitch_adj_deg}, {roll_adj_deg}) 度")
    
    # 复制并修改
    euler_angles_new_rad = euler_angles_rad.copy()
    euler_angles_new_rad[0] += math.radians(yaw_adj_deg)    # Yaw
    euler_angles_new_rad[1] += math.radians(pitch_adj_deg)  # Pitch
    euler_angles_new_rad[2] += math.radians(roll_adj_deg)   # Roll
    
    print(f"修改后欧拉角 (yaw, pitch, roll) in degrees: {np.rad2deg(euler_angles_new_rad)}")
    print("-" * 30)

    # --- 4. 欧拉角 -> 四元数 ---
    r_new = R.from_euler('zyx', euler_angles_new_rad)
    quat_xyzw_new = r_new.as_quat()

    # --- 5. 格式转换: [x, y, z, w] -> [w, x, y, z] ---
    quat_wxyz_new = np.array([quat_xyzw_new[3], quat_xyzw_new[0], quat_xyzw_new[1], quat_xyzw_new[2]])
    
    print(f"调整后新四元数 (w, x, y, z): {quat_wxyz_new}")
    return quat_wxyz_new

# =================== 参数调整区域 ===================
# 在这里修改你想要调整的角度值 (单位: 度)
# 正值表示增加, 负值表示减少

yaw_adjustment_degrees = 0.0      # 偏航角 (左右摇头)
pitch_adjustment_degrees = -18.0   # 俯仰角 (上下点头)
roll_adjustment_degrees = 0.0     # 滚转角 (歪头)

# =====================================================

# 初始四元数 [w, x, y, z]
initial_quat_wxyz = np.array([0.65309797, 0.27104066, 0.27104066, 0.65309797])

# 调用函数执行调整
new_quat = adjust_quaternion_by_euler(
    initial_quat_wxyz,
    yaw_adj_deg=yaw_adjustment_degrees,
    pitch_adj_deg=pitch_adjustment_degrees,
    roll_adj_deg=roll_adjustment_degrees
)