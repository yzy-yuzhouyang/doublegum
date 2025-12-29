import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from typing import Optional

# 假设你的 wrappers 文件夹下有这些工具类
import wrappers

def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             terminate_when_unhealthy: bool = True,
             action_concat: int = 1, # 保留参数接口以免报错，虽然逻辑可能被精简
             obs_concat: int = 1,
             continuous: bool = True,
             ) -> gym.Env:

    # 1. 获取当前 Gymnasium 注册的环境列表
    env_ids = list(gym.envs.registry.keys())

    # 2. 定义 MuJoCo Gym 环境白名单
    # 只要名字在列表里，或者包含 'Humanoid-v' 等特征，优先走 Gym 逻辑
    gym_mujoco_envs = [
        "Humanoid-v2", "Humanoid-v3", "Humanoid-v4",
        "Ant-v2", "Ant-v3", "Ant-v4",
        "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4",
        "Swimmer-v4"
    ]

    # ================= 核心逻辑：Gym vs DMC =================

    # 判定逻辑：如果在 Gym 注册表中，或者是我们熟知的 MuJoCo 环境名，则用 Gym 加载
    is_gym_env = (env_name in env_ids) or \
                 (env_name in gym_mujoco_envs)

    if is_gym_env:
        try:
            print(f"Loading Gym MuJoCo environment: {env_name}")
            env = gym.make(env_name)
        except (gym.error.Error, KeyError):
            # 自动降级/升级策略：处理 v3/v4 版本不匹配问题
            if "v3" in env_name:
                alt_name = env_name.replace("v3", "v4")
                print(f"[Warning] {env_name} not found in registry. Trying fallback to {alt_name}.")
                env = gym.make(alt_name)
            elif "v4" in env_name:
                alt_name = env_name.replace("v4", "v3")
                print(f"[Warning] {env_name} not found in registry. Trying fallback to {alt_name}.")
                env = gym.make(alt_name)
            else:
                raise

        save_folder = None

    else:
        # 否则，默认为 DeepMind Control Suite (DMC)
        print(f"Loading DMC environment: {env_name}")
        if '-' in env_name:
            domain_name, task_name = env_name.split('-')
        else:
            # 简单的容错，假设只有一个名字时是 domain，默认 task 为 walk
            domain_name = env_name
            task_name = 'walk'
            
        env = wrappers.DMCEnv(
            domain_name=domain_name, 
            task_name=task_name, 
            task_kwargs={'random': seed}
        )

    # ================= 通用 Wrappers 处理 =================

    # 1. 展平观测空间 (DMC 返回 Dict, Gym 有时返回 Dict)
    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
        env = wrappers.FlattenAction(env)

    # 2. 精度统一 (确保 Gym 环境也是 float32)
    if isinstance(env.observation_space, gym.spaces.Box):
         env = wrappers.SinglePrecision(env)

    # 3. 动作重复 (Frame Skip)
    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    # 4. 动作范围归一化 [-1, 1]
    if continuous:
        env = RescaleAction(env, -1.0, 1.0)

    # 5. 像素观测处理
    if from_pixels:
        if is_gym_env:
            camera_id = 0
        else:
            # DMC Quadruped 特殊处理
            camera_id = 2 if 'quadruped' in env_name else 0
            
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)

    # 6. 帧堆叠
    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    # 7. Sticky Action (可选)
    if sticky:
        env = wrappers.StickyActionEnv(env)

    # 8. 设置随机种子
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env