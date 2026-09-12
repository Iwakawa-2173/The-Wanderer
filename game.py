"""
Бесконечная мегаструктура на raymarching + SDF.

Управление:
  - WASD      — движение
  - Shift     — бег
  - Space     — прыжок
  - Ctrl      — переворот гравитации
  - F11       — полный экран
  - Esc       — выход (схлопывание + вспышка)

Звук:
  - step.mp3, falling.mp3, veter.mp3, quote.mp3 — лежат в папке `sounds`
    рядом с игрой. Работает и как .py, и как .exe (PyInstaller).
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import moderngl
import numpy as np
import pygame


# ─────────────────────────────────────────────────────────────────────────────
#  ПУТИ К РЕСУРСАМ
# ─────────────────────────────────────────────────────────────────────────────

def _resource_path(relative: str) -> str:
    """
    Путь к ресурсу. Работает:
      - как .py  → папка рядом с game.py
      - как .exe → временная папка PyInstaller (_MEIPASS)
    """
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative)


SOUND_DIR = _resource_path("sounds")


# ─────────────────────────────────────────────────────────────────────────────
#  КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_TITLE = "The Wanderer"
FULLSCREEN = True
TARGET_FPS = 60
MAX_DELTA_TIME = 0.05

CELL_SIZE = np.array([15.95, 7.95, 15.95], dtype=np.float64)
PLATE_HALF_THICKNESS = 0.15
HALL_HOLE_HALF_SIZE = 5.5
COLUMN_HALF_SIZE = np.array([1.8, 4.0, 1.8], dtype=np.float64)
FLOOR_HALF_EXTENT = 8.0

PLAYER_HEIGHT = 1.5
PLAYER_WALK_SPEED = 5.0
PLAYER_RUN_SPEED = 9.0
PLAYER_START_POS = np.array([0.0, 1.25, 10.0], dtype=np.float64)
PLAYER_START_YAW = 0.0
PLAYER_START_PITCH = 0.0

GRAVITY = 15.0
TERMINAL_VELOCITY = -20.0
JUMP_SPEED = 7.0
GROUND_CHECK_DIST = 0.15
MAX_PUSH_PER_FRAME = 0.5
MAX_WALL_PENETRATION = 0.3
COYOTE_TIME = 0.12
JUMP_BUFFER_TIME = 0.12

STAMINA_DURATION = 8.0
STAMINA_RECOVERY = 8.0
STAMINA_REST_DELAY = 1.2
STAMINA_MIN_TO_START = 0.15
SPEED_ACCEL_TIME = 0.6
SPEED_DECEL_TIME = 0.8

GRAVITY_FLIP_DURATION = 0.4
GRAVITY_FLIP_ROLL = math.pi

MOUSE_SENSITIVITY = 0.002
MAX_PITCH = math.pi / 2.05

RAY_STEPS = 256
RAY_MAX_DISTANCE = 200.0
RAY_EPSILON_BASE = 0.0015

# Заставка
INTRO_DURATION = 5.0
INTRO_FADE_TIME = 1.0
INTRO_CAMERA_SPIN = 0.08

# Цитаты — фазы
QUOTE_DIM_IN = 0.6
QUOTE_TEXT_IN = 0.8
QUOTE_HOLD = 5.0
QUOTE_FADE_OUT = 1.2
QUOTE_DIM_ALPHA = 1.0

# Выход
EXIT_COLLAPSE_TIME = 0.4
EXIT_FLASH_TIME = 0.15
EXIT_HOLD_WHITE_TIME = 0.15
EXIT_FOV_ZOOM = 10.0

# Звук — общие параметры
STEP_INTERVAL_WALK = 0.45
STEP_INTERVAL_RUN = 0.28
STEP_VOLUME_WALK = 0.55
STEP_VOLUME_RUN = 0.70
STEP_LANDING_VOLUME = 0.75
FALL_VOLUME_MAX = 0.7
FALL_MIN_SPEED = 8.0
FALL_RISE_SPEED = 3.0
FALL_FADE_SPEED = 5.0
WIND_MIN_DELAY = 8.0
WIND_MAX_DELAY = 25.0
WIND_VOLUME = 0.25
SOUND_DEBUG = False

# Тестовые цитаты
TEST_QUOTES: List[str] = [
    "Будучи завлечёнными внешним, люди часто забывают о внутреннем...",
    "То, за чем мы гонимся на протяжении вечности, на самом деле скрывается у нас за спиной...",
    "Всего лишь, нужно иметь скромность и мужество обернуться...",
    "Долг — это бесконечность всего возможного...",
    "Грубая сила — статистическая погрешность, когда дело заходит о вечности...",
    "Всё едино, всё связано, но это не значит, что ничего нет...",
    "Выход не снаружи — выход в восприятии...",
    "...нет более страшной тюрьмы, чем та, что мы создаём для себя сами...",
]


# ─────────────────────────────────────────────────────────────────────────────
#  ШЕЙДЕРЫ
# ─────────────────────────────────────────────────────────────────────────────

class Shaders:
    VERTEX = """
    #version 330 core
    in vec2 in_vert;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
    }
    """

    FRAGMENT_TEMPLATE = """
    #version 330 core

    uniform vec2 u_resolution;
    uniform vec3 u_player_pos;
    uniform vec3 u_forward;
    uniform vec3 u_right;
    uniform vec3 u_up;

    uniform sampler2D u_overlay_texture;
    uniform sampler2D u_dim_texture;
    uniform float u_overlay_alpha;
    uniform float u_dim_alpha;

    uniform float u_collapse;
    uniform float u_flash;

    out vec4 fragColor;

    const vec3  CELL              = vec3({cell_x}, {cell_y}, {cell_z});
    const float PLATE_THICKNESS   = {plate_thickness};
    const float HALL_HOLE_HALF    = {hall_hole_half};
    const vec3  COLUMN_HALF       = vec3({col_x}, {col_y}, {col_z});
    const float FLOOR_HALF_EXTENT = {floor_half_extent};
    const float COLLAPSE_ZOOM     = {collapse_zoom};

    float sdfBox(vec3 p, vec3 b) {{
        vec3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    }}

    float sdfFloors(vec3 p) {{
        float plate_y = mod(p.y + CELL.y * 0.5, CELL.y);
        float dist_y = abs(plate_y - CELL.y * 0.5) - PLATE_THICKNESS;

        vec2 cell_xz = p.xz - CELL.xz * floor(p.xz / CELL.xz + 0.5);
        float plate_xz = max(abs(cell_xz.x) - FLOOR_HALF_EXTENT,
                             abs(cell_xz.y) - FLOOR_HALF_EXTENT);

        float plate = max(dist_y, plate_xz);

        vec3 cell_local = vec3(cell_xz.x,
                               p.y - CELL.y * floor(p.y / CELL.y + 0.5),
                               cell_xz.y);
        float hole = sdfBox(cell_local, vec3(HALL_HOLE_HALF, CELL.y * 0.5, HALL_HOLE_HALF));

        return max(-hole, plate);
    }}

    float sdfColumns(vec3 p) {{
        vec3 cell = p - CELL * floor(p / CELL + 0.5);
        return sdfBox(cell, COLUMN_HALF);
    }}

    float map(vec3 p) {{
        return min(sdfColumns(p), sdfFloors(p));
    }}

    vec3 getNormal(vec3 p) {{
        vec2 e = vec2(0.002, 0.0);
        return normalize(vec3(
            map(p + e.xyy) - map(p - e.xyy),
            map(p + e.yxy) - map(p - e.yxy),
            map(p + e.yyx) - map(p - e.yyx)
        ));
    }}

    void main() {{
        vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;

        // Инвертированный зум: FOV расширяется, картинка уходит к краям.
        float fov_scale = 1.0 + u_collapse * (COLLAPSE_ZOOM - 1.0);
        vec3 rayDir = normalize(u_forward
                              + uv.x * u_right * fov_scale
                              + uv.y * u_up * fov_scale);

        float t = 0.02;
        bool hit = false;
        vec3 p;

        for (int i = 0; i < {ray_steps}; i++) {{
            p = u_player_pos + rayDir * t;
            float d = map(p);
            float eps = {ray_epsilon} * max(1.0, t);
            if (d < eps) {{ hit = true; break; }}
            t += max(d, 0.005 * t);
            if (t > {ray_max_dist}) break;
        }}

        vec3 color;
        if (hit) {{
            vec3 normal = getNormal(p);
            vec3 lightDir = normalize(u_player_pos - p);
            float diff = max(dot(normal, lightDir), 0.0);
            float dist = length(u_player_pos - p);
            float attenuation = 1.0 / (1.0 + 0.04 * dist + 0.015 * dist * dist);
            float fog = exp(-0.05 * t);

            vec3 world_amber = vec3(0.95, 0.70, 0.40);
            color = world_amber * (diff * 0.75 + 0.25) * attenuation * fog;
        }} else {{
            color = vec3(0.002);
        }}

        vec2 tex_uv = gl_FragCoord.xy / u_resolution;
        tex_uv.y = 1.0 - tex_uv.y;

        if (u_dim_alpha > 0.0) {{
            vec4 dim = texture(u_dim_texture, tex_uv);
            color = mix(color, dim.rgb, u_dim_alpha);
        }}

        if (u_overlay_alpha > 0.0) {{
            vec4 overlay = texture(u_overlay_texture, tex_uv);
            color = mix(color, overlay.rgb, overlay.a * u_overlay_alpha);
        }}

        if (u_flash > 0.0) {{
            color = mix(color, vec3(1.0), u_flash);
        }}

        fragColor = vec4(color, 1.0);
    }}
    """

    @classmethod
    def fragment(cls) -> str:
        return cls.FRAGMENT_TEMPLATE.format(
            cell_x=CELL_SIZE[0], cell_y=CELL_SIZE[1], cell_z=CELL_SIZE[2],
            plate_thickness=f"{PLATE_HALF_THICKNESS:.4f}",
            hall_hole_half=f"{HALL_HOLE_HALF_SIZE:.4f}",
            col_x=f"{COLUMN_HALF_SIZE[0]:.4f}",
            col_y=f"{COLUMN_HALF_SIZE[1]:.4f}",
            col_z=f"{COLUMN_HALF_SIZE[2]:.4f}",
            floor_half_extent=f"{FLOOR_HALF_EXTENT:.4f}",
            collapse_zoom=f"{EXIT_FOV_ZOOM:.4f}",
            ray_steps=RAY_STEPS,
            ray_max_dist=f"{RAY_MAX_DISTANCE:.1f}",
            ray_epsilon=f"{RAY_EPSILON_BASE:.4f}",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SDF-МОДЕЛЬ
# ─────────────────────────────────────────────────────────────────────────────

class SDFModel:
    CELL = CELL_SIZE
    PLATE_THICKNESS = PLATE_HALF_THICKNESS
    HALL_HOLE_HALF = HALL_HOLE_HALF_SIZE
    COLUMN_HALF = COLUMN_HALF_SIZE
    FLOOR_HALF_EXTENT = FLOOR_HALF_EXTENT

    @staticmethod
    def box(p: np.ndarray, b: np.ndarray) -> float:
        q = np.abs(p) - b
        return float(
            np.linalg.norm(np.maximum(q, 0.0))
            + min(max(q[0], max(q[1], q[2])), 0.0)
        )

    @classmethod
    def floors(cls, p: np.ndarray) -> float:
        plate_y = (p[1] + cls.CELL[1] * 0.5) % cls.CELL[1]
        dist_y = abs(plate_y - cls.CELL[1] * 0.5) - cls.PLATE_THICKNESS

        cell_xz = p[[0, 2]] - cls.CELL[[0, 2]] * np.floor(p[[0, 2]] / cls.CELL[[0, 2]] + 0.5)
        plate_xz = max(abs(cell_xz[0]) - cls.FLOOR_HALF_EXTENT,
                       abs(cell_xz[1]) - cls.FLOOR_HALF_EXTENT)

        plate = max(dist_y, plate_xz)

        cell_local = np.array([
            cell_xz[0],
            p[1] - cls.CELL[1] * math.floor(p[1] / cls.CELL[1] + 0.5),
            cell_xz[1],
        ])
        hole = cls.box(cell_local, np.array([
            cls.HALL_HOLE_HALF, cls.CELL[1] * 0.5, cls.HALL_HOLE_HALF
        ]))

        return max(-hole, plate)

    @classmethod
    def columns(cls, p: np.ndarray) -> float:
        cell = p - cls.CELL * np.floor(p / cls.CELL + 0.5)
        return cls.box(cell, cls.COLUMN_HALF)

    @classmethod
    def evaluate(cls, p: np.ndarray) -> float:
        return min(cls.columns(p), cls.floors(p))

    @classmethod
    def normal(cls, p: np.ndarray, eps: float = 0.002) -> np.ndarray:
        dx = cls.evaluate(p + np.array([eps, 0.0, 0.0])) \
           - cls.evaluate(p - np.array([eps, 0.0, 0.0]))
        dy = cls.evaluate(p + np.array([0.0, eps, 0.0])) \
           - cls.evaluate(p - np.array([0.0, eps, 0.0]))
        dz = cls.evaluate(p + np.array([0.0, 0.0, eps])) \
           - cls.evaluate(p - np.array([0.0, 0.0, eps]))
        n = np.array([dx, dy, dz], dtype=np.float64)
        length = float(np.linalg.norm(n))
        if length < 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return n / length

    @classmethod
    def distance_under_feet(cls, p: np.ndarray, sign: float = 1.0) -> float:
        return cls.evaluate(p + np.array([0.0, -PLAYER_HEIGHT * sign, 0.0]))


# ─────────────────────────────────────────────────────────────────────────────
#  ЗВУК
# ─────────────────────────────────────────────────────────────────────────────

class SoundManager:
    def __init__(self) -> None:
        try:
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
        except pygame.error as e:
            print(f"[SoundManager] Микшер недоступен: {e}")
            self.enabled = False
            return

        self.enabled = True

        self.step = self._load("step.mp3")
        self.falling = self._load("falling.mp3")
        self.wind = self._load("veter.mp3")
        self.quote = self._load("quote.mp3")

        self.falling_channel = None
        if self.falling is not None:
            self.falling_channel = self.falling.play(loops=-1)
            if self.falling_channel is not None:
                self.falling_channel.set_volume(0.0)

        self._step_timer = 0.0
        self._wind_timer = random.uniform(WIND_MIN_DELAY, WIND_MAX_DELAY)
        self._falling_volume = 0.0

    @staticmethod
    def _load(name: str):
        path = os.path.join(SOUND_DIR, name)
        try:
            return pygame.mixer.Sound(path)
        except (pygame.error, FileNotFoundError) as e:
            print(f"[SoundManager] {name} не загружен: {e}")
            return None

    def try_step(self, dt, is_moving, on_ground, run_intensity):
        if not self.enabled or self.step is None:
            return
        if on_ground and is_moving:
            self._step_timer -= dt
            if self._step_timer <= 0.0:
                ch = self.step.play()
                if ch is not None:
                    interval = (STEP_INTERVAL_WALK
                                + (STEP_INTERVAL_RUN - STEP_INTERVAL_WALK) * run_intensity)
                    vol = (STEP_VOLUME_WALK
                           + (STEP_VOLUME_RUN - STEP_VOLUME_WALK) * run_intensity)
                    ch.set_volume(vol)
                    self._step_timer = interval
        else:
            self._step_timer = 0.0

    def play_landing(self):
        if not self.enabled or self.step is None:
            return
        ch = self.step.play()
        if ch is not None:
            ch.set_volume(STEP_LANDING_VOLUME)

    def play_quote(self):
        if not self.enabled or self.quote is None:
            return
        self.quote.play()

    def update_falling(self, dt, y_velocity, on_ground):
        if not self.enabled or self.falling is None or self.falling_channel is None:
            return
        speed = abs(y_velocity)
        if on_ground or speed < FALL_MIN_SPEED:
            target = 0.0
        else:
            t = min(1.0, (speed - FALL_MIN_SPEED) /
                    (abs(TERMINAL_VELOCITY) - FALL_MIN_SPEED))
            target = t * FALL_VOLUME_MAX
        if target > self._falling_volume:
            self._falling_volume = min(target, self._falling_volume + dt * FALL_RISE_SPEED)
        else:
            self._falling_volume = max(target, self._falling_volume - dt * FALL_FADE_SPEED)
        self.falling_channel.set_volume(self._falling_volume)

    def update_wind(self, dt):
        if not self.enabled or self.wind is None:
            return
        self._wind_timer -= dt
        if self._wind_timer <= 0.0:
            ch = self.wind.play()
            if ch is not None:
                ch.set_volume(WIND_VOLUME)
            self._wind_timer = random.uniform(WIND_MIN_DELAY, WIND_MAX_DELAY)

    def update(self, dt, *, is_moving, run_intensity, on_ground,
               y_velocity, just_landed):
        if not just_landed:
            self.try_step(dt, is_moving, on_ground, run_intensity)
        self.update_falling(dt, y_velocity, on_ground)
        self.update_wind(dt)
        if just_landed:
            self.play_landing()

    def silence_falling(self):
        if self.falling_channel is not None:
            self._falling_volume = 0.0
            self.falling_channel.set_volume(0.0)

    def silence_all(self):
        if self.falling_channel is not None:
            self.falling_channel.set_volume(0.0)
        try:
            pygame.mixer.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  ГРАВИТАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GravityState:
    sign: float = 1.0
    target_sign: float = 1.0
    roll: float = 0.0
    target_roll: float = 0.0
    flip_angle_switched: bool = False

    @property
    def transitioning(self) -> bool:
        return abs(self.roll - self.target_roll) > 1e-4

    def request_flip(self) -> None:
        if self.transitioning:
            return
        self.target_sign = -self.sign
        self.target_roll = self.roll + GRAVITY_FLIP_ROLL
        self.flip_angle_switched = False

    def update(self, dt: float) -> None:
        if not self.transitioning:
            return
        speed = GRAVITY_FLIP_ROLL / GRAVITY_FLIP_DURATION
        diff = self.target_roll - self.roll
        step = min(abs(diff), speed * dt) * (1.0 if diff > 0 else -1.0)
        self.roll += step
        if not self.flip_angle_switched:
            elapsed = abs(self.roll - (self.target_roll - GRAVITY_FLIP_ROLL))
            if elapsed >= GRAVITY_FLIP_ROLL * 0.5:
                self.sign = self.target_sign
                self.flip_angle_switched = True
        if abs(self.roll - self.target_roll) < 1e-4:
            self.roll = self.target_roll
            self.roll = self.roll % (2.0 * math.pi)
            self.target_roll = self.roll


# ─────────────────────────────────────────────────────────────────────────────
#  СТАМИНА
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stamina:
    value: float = 1.0
    rest_timer: float = 0.0
    exhausted: bool = False
    speed_factor: float = 0.0

    def is_active(self) -> bool:
        return not self.exhausted and self.value > 0.0

    def update(self, dt, wants_to_run, on_ground, is_moving):
        actually_running = (
            wants_to_run and on_ground and is_moving and self.is_active()
        )
        if actually_running:
            self.value -= dt / STAMINA_DURATION
            self.rest_timer = STAMINA_REST_DELAY
            if self.value <= 0.0:
                self.value = 0.0
                self.exhausted = True
        else:
            if self.rest_timer > 0.0:
                self.rest_timer -= dt
            else:
                self.value += dt / STAMINA_RECOVERY
                if self.value >= 1.0:
                    self.value = 1.0
            if self.exhausted and self.value >= STAMINA_MIN_TO_START:
                self.exhausted = False

        if (wants_to_run and on_ground and is_moving and self.is_active()):
            target = 1.0
        else:
            target = 0.0
        if target > self.speed_factor:
            self.speed_factor = min(target, self.speed_factor + dt / SPEED_ACCEL_TIME)
        else:
            self.speed_factor = max(target, self.speed_factor - dt / SPEED_DECEL_TIME)

    def movement_speed(self) -> float:
        return (PLAYER_WALK_SPEED
                + (PLAYER_RUN_SPEED - PLAYER_WALK_SPEED) * self.speed_factor)


# ─────────────────────────────────────────────────────────────────────────────
#  КАМЕРА
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    yaw: float = PLAYER_START_YAW
    pitch: float = PLAYER_START_PITCH

    def rotate(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch = max(-MAX_PITCH, min(MAX_PITCH, self.pitch + dpitch))

    @property
    def forward(self):
        cp = math.cos(self.pitch)
        return np.array([
            cp * math.sin(self.yaw),
            math.sin(self.pitch),
            cp * math.cos(self.yaw),
        ], dtype=np.float64)

    @property
    def right(self):
        return np.array([
            math.cos(self.yaw),
            0.0,
            -math.sin(self.yaw),
        ], dtype=np.float64)

    def rolled_right(self, roll):
        r = self.right
        u = np.cross(self.forward, r).astype(np.float64)
        if abs(roll) > 1e-9:
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            r = r * cos_r + u * sin_r
        return r

    def rolled_up(self, roll):
        f = self.forward
        r = self.right
        u = np.cross(f, r).astype(np.float64)
        if abs(roll) > 1e-9:
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            u = r * (-sin_r) + u * cos_r
        return u

    @property
    def flat_forward(self):
        f = np.array([self.forward[0], 0.0, self.forward[2]], dtype=np.float64)
        n = float(np.linalg.norm(f))
        return f / n if n > 1e-9 else f


# ─────────────────────────────────────────────────────────────────────────────
#  ВВОД
# ─────────────────────────────────────────────────────────────────────────────

class InputController:
    def __init__(self) -> None:
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    @staticmethod
    def poll_events() -> Tuple[bool, bool, bool, bool, bool]:
        quit_requested = False
        jump_pressed = False
        flip_pressed = False
        toggle_fullscreen = False
        any_key = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                any_key = True
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
                elif event.key == pygame.K_SPACE:
                    jump_pressed = True
                elif event.key in (pygame.K_LCTRL, pygame.K_RCTRL):
                    flip_pressed = True
                elif event.key == pygame.K_F11:
                    toggle_fullscreen = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                any_key = True
        return quit_requested, jump_pressed, flip_pressed, \
               toggle_fullscreen, any_key

    @staticmethod
    def mouse_delta():
        dx, dy = pygame.mouse.get_rel()
        return dx * MOUSE_SENSITIVITY, dy * MOUSE_SENSITIVITY

    @staticmethod
    def movement_axes():
        keys = pygame.key.get_pressed()
        fwd = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        rgt = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        return fwd, rgt

    @staticmethod
    def wants_to_run() -> bool:
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])


# ─────────────────────────────────────────────────────────────────────────────
#  ИГРОК
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Player:
    position: np.ndarray = field(default_factory=lambda: PLAYER_START_POS.copy())
    y_velocity: float = 0.0
    on_ground: bool = False
    coyote_timer: float = 0.0
    jump_buffer: float = 0.0
    just_landed: bool = False

    def request_jump(self):
        self.jump_buffer = JUMP_BUFFER_TIME

    def update(self, camera, gravity, move_fwd, move_right, speed, dt):
        self._update_timers(dt)
        self._try_jump(gravity)
        self._apply_movement(camera, move_fwd, move_right, speed, dt)
        self._apply_gravity(gravity, dt)
        self._resolve_collisions(gravity)

    def _update_timers(self, dt):
        if self.on_ground:
            self.coyote_timer = COYOTE_TIME
        else:
            self.coyote_timer -= dt
        if self.jump_buffer > 0.0:
            self.jump_buffer -= dt

    def _try_jump(self, gravity):
        if self.jump_buffer > 0.0 and self.coyote_timer > 0.0:
            self.y_velocity = JUMP_SPEED * gravity.sign
            self.jump_buffer = 0.0
            self.coyote_timer = 0.0
            self.on_ground = False

    def _apply_movement(self, camera, move_fwd, move_right, speed, dt):
        delta = (camera.flat_forward * move_fwd + camera.right * move_right) * speed * dt
        self.position = self.position + delta

    def _apply_gravity(self, gravity, dt):
        self.y_velocity -= GRAVITY * gravity.sign * dt
        if gravity.sign > 0 and self.y_velocity < TERMINAL_VELOCITY:
            self.y_velocity = TERMINAL_VELOCITY
        elif gravity.sign < 0 and self.y_velocity > -TERMINAL_VELOCITY:
            self.y_velocity = -TERMINAL_VELOCITY
        self.position[1] += self.y_velocity * dt

    def _resolve_collisions(self, gravity):
        was_on_ground = self.on_ground
        self.just_landed = False

        dist_under = SDFModel.distance_under_feet(self.position, gravity.sign)
        moving_toward_ground = (
            (gravity.sign > 0 and self.y_velocity <= 0.0) or
            (gravity.sign < 0 and self.y_velocity >= 0.0)
        )

        if dist_under < GROUND_CHECK_DIST and moving_toward_ground:
            push = min(GROUND_CHECK_DIST - dist_under, MAX_PUSH_PER_FRAME)
            self.position[1] += push * gravity.sign
            if not was_on_ground:
                self.just_landed = True
            self.y_velocity = 0.0
            self.on_ground = True
        else:
            self.on_ground = False

        dist = SDFModel.evaluate(self.position)
        if dist > MAX_WALL_PENETRATION:
            return
        normal = SDFModel.normal(self.position)
        push = MAX_WALL_PENETRATION - dist
        self.position = self.position + normal * push
        if normal[1] * gravity.sign > 0.3:
            self.y_velocity = 0.0
            self.on_ground = True


# ─────────────────────────────────────────────────────────────────────────────
#  ЗАСТАВКА
# ─────────────────────────────────────────────────────────────────────────────

class IntroScreen:
    def __init__(self, ctx, width, height, duration=INTRO_DURATION):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.duration = duration
        self.elapsed = 0.0

        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 255))
        self._draw_text(surface)

        data = pygame.image.tostring(surface, "RGBA", False)
        self.texture = ctx.texture((width, height), 4, data)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _draw_text(self, surface):
        f_title = pygame.font.SysFont("consolas,dejavusansmono,monospace", 64, bold=True)
        f_sub = pygame.font.SysFont("consolas,dejavusansmono,monospace", 26)
        f_body = pygame.font.SysFont("consolas,dejavusansmono,monospace", 22)
        f_hint = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18)

        c_title = (255, 200, 130)
        c_body = (200, 180, 150)
        c_hint = (120, 110, 100)

        title = f_title.render("The Wanderer", True, c_title)
        surface.blit(title, title.get_rect(center=(self.width // 2, self.height // 3 - 30)))

        sub = f_sub.render("Скиталец", True, c_body)
        surface.blit(sub, sub.get_rect(center=(self.width // 2, self.height // 3 + 40)))

        sep_y = self.height // 3 + 90
        pygame.draw.line(surface, (80, 70, 60),
                         (self.width // 3, sep_y),
                         (self.width * 2 // 3, sep_y), 1)

        controls = [
            ("W A S D",  "движение"),
            ("Shift",    "бег"),
            ("Space",    "прыжок"),
            ("Ctrl",     "переворот гравитации"),
            ("F11",      "полный экран"),
            ("Esc",      "выход"),
        ]
        lh = 36
        sy = self.height // 2 + 20
        for i, (k, d) in enumerate(controls):
            kt = f_body.render(k, True, c_title)
            dt = f_body.render(d, True, c_body)
            y = sy + i * lh
            surface.blit(kt, kt.get_rect(midright=(self.width // 2 - 20, y)))
            surface.blit(dt, dt.get_rect(midleft=(self.width // 2 + 20, y)))

        hint = f_hint.render("нажмите любую клавишу, чтобы начать", True, c_hint)
        surface.blit(hint, hint.get_rect(center=(self.width // 2, self.height - 80)))

    def update(self, dt):
        self.elapsed += dt

    @property
    def alpha(self):
        if self.elapsed >= self.duration:
            return 0.0
        fade_start = self.duration - INTRO_FADE_TIME
        if self.elapsed < fade_start:
            return 1.0
        return 1.0 - (self.elapsed - fade_start) / INTRO_FADE_TIME

    @property
    def is_done(self):
        return self.elapsed >= self.duration

    def skip(self):
        self.elapsed = self.duration

    def release(self):
        try:
            self.texture.release()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  ЦИТАТА
# ─────────────────────────────────────────────────────────────────────────────

class QuoteScreen:
    PHASE_DIM_IN = 0
    PHASE_TEXT_IN = 1
    PHASE_HOLD = 2
    PHASE_FADE_OUT = 3
    PHASE_DONE = 4

    def __init__(self, ctx, width, height, text):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.text = text
        self.elapsed = 0.0
        self.phase = self.PHASE_DIM_IN

        black = pygame.Surface((1, 1), pygame.SRCALPHA)
        black.fill((0, 0, 0, 255))
        dim_data = pygame.image.tostring(black, "RGBA", False)
        self.dim_texture = ctx.texture((1, 1), 4, dim_data)
        self.dim_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))
        self._draw_quote(surface)

        data = pygame.image.tostring(surface, "RGBA", False)
        self.text_texture = ctx.texture((width, height), 4, data)
        self.text_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _draw_quote(self, surface):
        f_quote = pygame.font.SysFont(
            "georgia,timesnewroman,serif", 36, italic=True)
        color = (235, 220, 195)

        quoted = f"\u00ab{self.text}\u00bb"
        max_width = int(self.width * 0.7)
        lines = self._wrap_text(f_quote, quoted, max_width)

        lh = f_quote.get_height() + 8
        total = lh * len(lines)
        sy = (self.height - total) // 2

        for i, line in enumerate(lines):
            rendered = f_quote.render(line, True, color)
            rect = rendered.get_rect(
                center=(self.width // 2, sy + i * lh + lh // 2))
            surface.blit(rendered, rect)

    @staticmethod
    def _wrap_text(font, text, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def update(self, dt):
        if self.phase == self.PHASE_DONE:
            return
        self.elapsed += dt

        t1 = QUOTE_DIM_IN
        t2 = t1 + QUOTE_TEXT_IN
        t3 = t2 + QUOTE_HOLD
        t4 = t3 + QUOTE_FADE_OUT

        if self.phase == self.PHASE_DIM_IN and self.elapsed >= t1:
            self.phase = self.PHASE_TEXT_IN
        if self.phase == self.PHASE_TEXT_IN and self.elapsed >= t2:
            self.phase = self.PHASE_HOLD
        if self.phase == self.PHASE_HOLD and self.elapsed >= t3:
            self.phase = self.PHASE_FADE_OUT
        if self.phase == self.PHASE_FADE_OUT and self.elapsed >= t4:
            self.phase = self.PHASE_DONE

    @property
    def dim_alpha(self):
        if self.phase == self.PHASE_DIM_IN:
            t = min(1.0, self.elapsed / QUOTE_DIM_IN)
            return t * QUOTE_DIM_ALPHA
        if self.phase in (self.PHASE_TEXT_IN, self.PHASE_HOLD):
            return QUOTE_DIM_ALPHA
        if self.phase == self.PHASE_FADE_OUT:
            t = self.elapsed - (QUOTE_DIM_IN + QUOTE_TEXT_IN + QUOTE_HOLD)
            k = max(0.0, 1.0 - t / QUOTE_FADE_OUT)
            return k * QUOTE_DIM_ALPHA
        return 0.0

    @property
    def text_alpha(self):
        if self.phase == self.PHASE_DIM_IN:
            return 0.0
        if self.phase == self.PHASE_TEXT_IN:
            t = self.elapsed - QUOTE_DIM_IN
            return min(1.0, t / QUOTE_TEXT_IN)
        if self.phase == self.PHASE_HOLD:
            return 1.0
        if self.phase == self.PHASE_FADE_OUT:
            t = self.elapsed - (QUOTE_DIM_IN + QUOTE_TEXT_IN + QUOTE_HOLD)
            return max(0.0, 1.0 - t / QUOTE_FADE_OUT)
        return 0.0

    @property
    def is_done(self):
        return self.phase == self.PHASE_DONE

    def release(self):
        try:
            self.dim_texture.release()
        except Exception:
            pass
        try:
            self.text_texture.release()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  ЭФФЕКТ ВЫХОДА
# ─────────────────────────────────────────────────────────────────────────────

class ExitEffect:
    PHASE_COLLAPSE = 0
    PHASE_FLASH = 1
    PHASE_HOLD_WHITE = 2
    PHASE_DONE = 3

    def __init__(self,
                 collapse_time=EXIT_COLLAPSE_TIME,
                 flash_time=EXIT_FLASH_TIME,
                 hold_time=EXIT_HOLD_WHITE_TIME):
        self.collapse_time = collapse_time
        self.flash_time = flash_time
        self.hold_time = hold_time
        self.elapsed = 0.0
        self.phase = self.PHASE_COLLAPSE

    def update(self, dt):
        if self.phase == self.PHASE_DONE:
            return
        self.elapsed += dt

        if self.phase == self.PHASE_COLLAPSE:
            if self.elapsed >= self.collapse_time:
                self.phase = self.PHASE_FLASH
                self.elapsed = 0.0
        elif self.phase == self.PHASE_FLASH:
            if self.elapsed >= self.flash_time:
                self.phase = self.PHASE_HOLD_WHITE
                self.elapsed = 0.0
        elif self.phase == self.PHASE_HOLD_WHITE:
            if self.elapsed >= self.hold_time:
                self.phase = self.PHASE_DONE

    @property
    def collapse(self):
        if self.phase == self.PHASE_COLLAPSE:
            return min(1.0, self.elapsed / self.collapse_time)
        return 1.0

    @property
    def flash(self):
        if self.phase == self.PHASE_COLLAPSE:
            return 0.0
        if self.phase == self.PHASE_FLASH:
            return min(1.0, self.elapsed / self.flash_time)
        return 1.0

    @property
    def is_done(self):
        return self.phase == self.PHASE_DONE


# ─────────────────────────────────────────────────────────────────────────────
#  ПУЛ ЦИТАТ (без повторов)
# ─────────────────────────────────────────────────────────────────────────────

class QuotePool:
    def __init__(self, quotes: List[str]) -> None:
        self._all = list(quotes)
        self._remaining: List[str] = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._remaining = list(self._all)
        random.shuffle(self._remaining)

    def next(self) -> str:
        if not self._remaining:
            self._reshuffle()
        return self._remaining.pop()


# ─────────────────────────────────────────────────────────────────────────────
#  ТРИГГЕРЫ ЦИТАТ
# ─────────────────────────────────────────────────────────────────────────────

class QuoteTrigger:
    FALL_TRIGGER_DEPTH = 12.0
    RUN_TRIGGER_DISTANCE = 60.0
    QUOTE_COOLDOWN = 25.0

    def __init__(self, pool: QuotePool) -> None:
        self.pool = pool
        self.cooldown = 0.0
        self.flipped_once = False
        self.fall_start_y: Optional[float] = None
        self.fall_triggered = False
        self.run_distance = 0.0

    def update(self, dt, player, gravity, stamina) -> Optional[str]:
        if self.cooldown > 0.0:
            self.cooldown -= dt

        if player.on_ground:
            self.fall_start_y = player.position[1]
            self.fall_triggered = False
        else:
            if self.fall_start_y is None:
                self.fall_start_y = player.position[1]
            depth = abs(self.fall_start_y - player.position[1])
            if (not self.fall_triggered
                    and depth > self.FALL_TRIGGER_DEPTH
                    and self.cooldown <= 0.0):
                self.fall_triggered = True
                self.cooldown = self.QUOTE_COOLDOWN
                return self.pool.next()

        if player.on_ground and stamina.speed_factor > 0.5:
            self.run_distance += stamina.movement_speed() * dt
            if (self.run_distance > self.RUN_TRIGGER_DISTANCE
                    and self.cooldown <= 0.0):
                self.run_distance = 0.0
                self.cooldown = self.QUOTE_COOLDOWN
                return self.pool.next()

        return None

    def on_gravity_flip(self) -> Optional[str]:
        if self.flipped_once or self.cooldown > 0.0:
            return None
        self.flipped_once = True
        self.cooldown = self.QUOTE_COOLDOWN
        return self.pool.next()


# ─────────────────────────────────────────────────────────────────────────────
#  РЕНДЕР
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=Shaders.VERTEX,
            fragment_shader=Shaders.fragment(),
        )
        vertices = np.array([
            -1.0, -1.0,   1.0, -1.0,  -1.0,  1.0,
            -1.0,  1.0,   1.0, -1.0,   1.0,  1.0,
        ], dtype='f4')
        self.vbo = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(self.program, [(self.vbo, '2f', 'in_vert')])

        self.overlay_texture = None
        self.dim_texture = None
        self.overlay_alpha = 0.0
        self.dim_alpha = 0.0

        self.program['u_overlay_texture'] = 0
        self.program['u_dim_texture'] = 1
        self.program['u_overlay_alpha'].value = 0.0
        self.program['u_dim_alpha'].value = 0.0
        self.program['u_collapse'].value = 0.0
        self.program['u_flash'].value = 0.0

    def set_resolution(self, width, height):
        self.program['u_resolution'].write(
            np.array([width, height], dtype='f4').tobytes()
        )

    def set_overlay(self, texture, alpha):
        self.overlay_texture = texture
        self.overlay_alpha = alpha

    def set_dim(self, texture, alpha):
        self.dim_texture = texture
        self.dim_alpha = alpha

    def set_exit_effect(self, collapse: float, flash: float) -> None:
        self.program['u_collapse'].value = collapse
        self.program['u_flash'].value = flash

    def draw(self, camera, gravity, player_pos):
        forward = camera.forward
        right = camera.rolled_right(gravity.roll)
        up = camera.rolled_up(gravity.roll)

        self.program['u_player_pos'].write(player_pos.astype('f4').tobytes())
        self.program['u_forward'].write(forward.astype('f4').tobytes())
        self.program['u_right'].write(right.astype('f4').tobytes())
        self.program['u_up'].write(up.astype('f4').tobytes())

        if self.dim_texture is not None and self.dim_alpha > 0.0:
            self.dim_texture.use(location=1)
            self.program['u_dim_alpha'].value = self.dim_alpha
        else:
            self.program['u_dim_alpha'].value = 0.0

        if self.overlay_texture is not None and self.overlay_alpha > 0.0:
            self.overlay_texture.use(location=0)
            self.program['u_overlay_alpha'].value = self.overlay_alpha
        else:
            self.program['u_overlay_alpha'].value = 0.0

        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render()


# ─────────────────────────────────────────────────────────────────────────────
#  ИГРА
# ─────────────────────────────────────────────────────────────────────────────

class Game:
    def __init__(self):
        pygame.init()

        self.fullscreen = FULLSCREEN
        self.window_width, self.window_height = self._create_display()
        pygame.display.set_caption(WINDOW_TITLE)

        self.clock = pygame.time.Clock()
        self.ctx = moderngl.create_context()

        self.camera = Camera()
        self.gravity = GravityState()
        self.stamina = Stamina()
        self.player = Player()
        self.input = InputController()
        self.renderer = Renderer(self.ctx)
        self.renderer.set_resolution(self.window_width, self.window_height)
        self.sound = SoundManager()

        self.intro = IntroScreen(self.ctx, self.window_width,
                                 self.window_height, INTRO_DURATION)

        self.quote_pool = QuotePool(TEST_QUOTES)
        self.quote_trigger = QuoteTrigger(self.quote_pool)
        self.active_quote: Optional[QuoteScreen] = None

        self.exit_effect: Optional[ExitEffect] = None

        self.running = True

    def _create_display(self):
        if self.fullscreen:
            info = pygame.display.Info()
            w, h = info.current_w, info.current_h
            pygame.display.set_mode(
                (w, h),
                pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN,
            )
        else:
            w, h = 1024, 768
            pygame.display.set_mode(
                (w, h),
                pygame.OPENGL | pygame.DOUBLEBUF,
            )
        return w, h

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.window_width, self.window_height = self._create_display()
        self.renderer.set_resolution(self.window_width, self.window_height)

        self.intro.release()
        self.intro = IntroScreen(self.ctx, self.window_width,
                                 self.window_height, self.intro.duration)

        if self.active_quote is not None and not self.active_quote.is_done:
            text = self.active_quote.text
            self.active_quote.release()
            self.active_quote = QuoteScreen(
                self.ctx, self.window_width, self.window_height, text)

        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def _start_quote(self, text):
        if self.active_quote is not None:
            self.active_quote.release()
        self.active_quote = QuoteScreen(
            self.ctx, self.window_width, self.window_height, text)
        self.sound.play_quote()

    def _release_quote(self):
        if self.active_quote is not None:
            self.active_quote.release()
            self.active_quote = None

    def _start_exit(self):
        if self.exit_effect is not None:
            return
        self.exit_effect = ExitEffect()
        self.sound.silence_all()

    def run(self):
        while self.running:
            dt = self._delta_time()
            self._process_input(dt)
            self._update(dt)
            self._render()
        pygame.display.flip()
        pygame.time.wait(100)
        pygame.quit()

    def _delta_time(self):
        dt = self.clock.tick(TARGET_FPS) / 1000.0
        return min(dt, MAX_DELTA_TIME)

    def _process_input(self, dt):
        quit_requested, jump_pressed, flip_pressed, toggle_fs, any_key = \
            self.input.poll_events()

        if self.exit_effect is not None:
            return

        if quit_requested:
            self._start_exit()
            return

        if toggle_fs:
            self._toggle_fullscreen()

        if not self.intro.is_done:
            if any_key:
                self.intro.skip()
            return

        if jump_pressed:
            self.player.request_jump()
        if flip_pressed:
            self.gravity.request_flip()
            quote = self.quote_trigger.on_gravity_flip()
            if quote is not None:
                self._start_quote(quote)

        dyaw, dpitch = self.input.mouse_delta()
        if math.cos(self.gravity.roll) < 0.0:
            dyaw = -dyaw
            dpitch = -dpitch
        self.camera.rotate(dyaw, -dpitch)

    def _update(self, dt):
        if self.exit_effect is not None:
            self.exit_effect.update(dt)
            if self.exit_effect.is_done:
                self.running = False
            return

        if not self.intro.is_done:
            self.intro.update(dt)
            self.camera.yaw += dt * INTRO_CAMERA_SPIN
            self.sound.silence_falling()
            return

        if self.active_quote is not None:
            self.active_quote.update(dt)
            if self.active_quote.is_done:
                self._release_quote()

        self.gravity.update(dt)

        move_fwd, move_right = self.input.movement_axes()
        wants_to_run = self.input.wants_to_run()
        is_moving = abs(move_fwd) > 0.01 or abs(move_right) > 0.01

        self.stamina.update(dt, wants_to_run=wants_to_run,
                           on_ground=self.player.on_ground,
                           is_moving=is_moving)

        speed = self.stamina.movement_speed()
        self.player.update(self.camera, self.gravity,
                           move_fwd, move_right, speed, dt)

        self.sound.update(dt, is_moving=is_moving,
                          run_intensity=self.stamina.speed_factor,
                          on_ground=self.player.on_ground,
                          y_velocity=self.player.y_velocity,
                          just_landed=self.player.just_landed)

        if self.active_quote is None:
            quote = self.quote_trigger.update(
                dt, self.player, self.gravity, self.stamina)
            if quote is not None:
                self._start_quote(quote)

    def _render(self):
        if self.active_quote is not None and not self.active_quote.is_done:
            self.renderer.set_dim(self.active_quote.dim_texture,
                                  self.active_quote.dim_alpha)
            self.renderer.set_overlay(self.active_quote.text_texture,
                                      self.active_quote.text_alpha)
        elif not self.intro.is_done:
            self.renderer.set_dim(None, 0.0)
            self.renderer.set_overlay(self.intro.texture, self.intro.alpha)
        else:
            self.renderer.set_dim(None, 0.0)
            self.renderer.set_overlay(None, 0.0)

        if self.exit_effect is not None:
            self.renderer.set_exit_effect(
                self.exit_effect.collapse,
                self.exit_effect.flash,
            )
        else:
            self.renderer.set_exit_effect(0.0, 0.0)

        self.renderer.draw(self.camera, self.gravity, self.player.position)
        pygame.display.flip()


def main():
    Game().run()


if __name__ == "__main__":
    main()