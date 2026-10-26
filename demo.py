"""Live demo: watch controllers handle the same traffic side by side.

    python demo.py                                        # Fixed-time vs Dueling DDQN, rush hour
    python demo.py --agents actuated dqn --scenario asymmetric
    python demo.py --agents fixed_time actuated dueling_ddqn --speed 40
    python demo.py --headless --until 600 --screenshot results/screenshots/rush_hour.png

Keys: SPACE pause/resume, UP/DOWN speed, R restart, S screenshot, ESC quit.
Every panel uses the same traffic seed, so all controllers face identical arrivals.
"""
import argparse
import os
import sys

import config
from agents import load_trained_agent
from plots import AGENT_COLORS
from traffic_env import PHASE_LANES, TrafficIntersectionEnv

PANEL_W, PANEL_H, INFO_H, TOP_H = 480, 480, 130, 44
ROAD = 64                      # road width in px (one lane each way)
CAR_LEN, CAR_WID, CAR_GAP = 12, 9, 3
STEP = CAR_LEN + CAR_GAP
STOP_OFFSET = 6                # gap between the stop line and the first car
MAX_VISIBLE = (PANEL_W // 2 - ROAD // 2 - STOP_OFFSET) // STEP
FPS = 30


def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


SURFACE = rgb("#fcfcfb")
INK = rgb("#0b0b0b")
INK_SECONDARY = rgb("#52514e")
MUTED = rgb("#898781")
AXIS = rgb("#c3c2b7")
GRASS = rgb("#e3e8dc")
ASPHALT = rgb("#55554f")
MARKING = rgb("#d9d7cc")
HOUSING = rgb("#2c2c2a")
LIGHT = {"green": rgb("#0ca30c"), "yellow": rgb("#fab219"), "red": rgb("#d03b3b")}
CAR_COLORS = [rgb(c) for c in ("#f4f3ee", "#2a78d6", "#e34948", "#b9b8b0",
                               "#eda100", "#3b3b38", "#1baf7a", "#9085e9")]


def light_state(env, lane):
    if env.yellow_left > 0:
        return "yellow" if lane in PHASE_LANES[1 - env.phase] else "red"
    return "green" if lane in PHASE_LANES[env.phase] else "red"


def car_rect(pygame, lane, k, cx, cy):
    """Rectangle of the k-th queued car (k=0 at the stop line). Right-hand traffic."""
    half, quarter = ROAD // 2, ROAD // 4
    if lane == 0:  # from the north, heading south, west half of the road
        bottom = cy - half - STOP_OFFSET - k * STEP
        return pygame.Rect(cx - quarter - CAR_WID // 2, bottom - CAR_LEN, CAR_WID, CAR_LEN)
    if lane == 1:  # from the south, heading north, east half
        top = cy + half + STOP_OFFSET + k * STEP
        return pygame.Rect(cx + quarter - CAR_WID // 2, top, CAR_WID, CAR_LEN)
    if lane == 2:  # from the east, heading west, north half
        left = cx + half + STOP_OFFSET + k * STEP
        return pygame.Rect(left, cy - quarter - CAR_WID // 2, CAR_LEN, CAR_WID)
    right = cx - half - STOP_OFFSET - k * STEP  # from the west, heading east, south half
    return pygame.Rect(right - CAR_LEN, cy + quarter - CAR_WID // 2, CAR_LEN, CAR_WID)


def draw_intersection(pygame, screen, fonts, ox, oy, env):
    cx, cy, half = ox + PANEL_W // 2, oy + PANEL_H // 2, ROAD // 2
    pygame.draw.rect(screen, GRASS, (ox, oy, PANEL_W, PANEL_H))
    pygame.draw.rect(screen, ASPHALT, (cx - half, oy, ROAD, PANEL_H))
    pygame.draw.rect(screen, ASPHALT, (ox, cy - half, PANEL_W, ROAD))

    # Dashed centre lines, stopping at the junction box.
    for y in list(range(oy + 4, cy - half - 8, 16)) + list(range(cy + half + 8, oy + PANEL_H - 8, 16)):
        pygame.draw.line(screen, MARKING, (cx, y), (cx, y + 8), 2)
    for x in list(range(ox + 4, cx - half - 8, 16)) + list(range(cx + half + 8, ox + PANEL_W - 8, 16)):
        pygame.draw.line(screen, MARKING, (x, cy), (x + 8, cy), 2)

    # Stop lines across each incoming half of the road.
    pygame.draw.line(screen, MARKING, (cx - half, cy - half - 3), (cx, cy - half - 3), 3)
    pygame.draw.line(screen, MARKING, (cx, cy + half + 3), (cx + half, cy + half + 3), 3)
    pygame.draw.line(screen, MARKING, (cx + half + 3, cy - half), (cx + half + 3, cy), 3)
    pygame.draw.line(screen, MARKING, (cx - half - 3, cy), (cx - half - 3, cy + half), 3)

    # Queued cars. A car's colour is tied to its arrival order, so the queue
    # visibly creeps forward as cars at the front leave.
    for lane in range(4):
        queue = int(env.queues[lane])
        visible = queue if queue <= MAX_VISIBLE else MAX_VISIBLE - 1
        for k in range(visible):
            car_id = int(env.departed_per_lane[lane]) + k
            color = CAR_COLORS[(car_id * 5 + lane * 3) % len(CAR_COLORS)]
            pygame.draw.rect(screen, color, car_rect(pygame, lane, k, cx, cy), border_radius=3)
        if queue > visible:
            slot = car_rect(pygame, lane, MAX_VISIBLE - 1, cx, cy)
            label = fonts["small_bold"].render(f"+{queue - visible}", True, SURFACE)
            box = label.get_rect(center=slot.center).inflate(8, 4)
            pygame.draw.rect(screen, HOUSING, box, border_radius=6)
            screen.blit(label, label.get_rect(center=box.center))

    # One signal head per approach, on the driver's right before the stop line.
    heads = {0: (cx - half - 12, cy - half - 12), 1: (cx + half + 12, cy + half + 12),
             2: (cx + half + 12, cy - half - 12), 3: (cx - half - 12, cy + half + 12)}
    for lane, center in heads.items():
        pygame.draw.circle(screen, HOUSING, center, 9)
        pygame.draw.circle(screen, LIGHT[light_state(env, lane)], center, 6)

    # Queue count next to each approach.
    for lane, (dx, dy) in {0: (-46, -70), 1: (46, 70), 2: (70, -46), 3: (-70, 46)}.items():
        text = fonts["small"].render(f"{int(env.queues[lane])}", True, INK_SECONDARY)
        screen.blit(text, text.get_rect(center=(cx + dx, cy + dy)))


def draw_info(pygame, screen, fonts, ox, oy, env, name):
    pygame.draw.rect(screen, SURFACE, (ox, oy, PANEL_W, INFO_H))
    pygame.draw.rect(screen, rgb(AGENT_COLORS[name]), (ox + 16, oy + 16, 12, 12), border_radius=3)
    screen.blit(fonts["bold"].render(config.AGENT_LABELS[name], True, INK), (ox + 36, oy + 11))
    m = env.metrics()
    stats = [("Average wait per vehicle", f"{m['avg_wait']:.1f} s"),
             ("Vehicles waiting now", f"{m['waiting']}"),
             ("Vehicles served", f"{m['throughput']}"),
             ("Phase switches", f"{m['switches']}")]
    for i, (label, value) in enumerate(stats):
        y = oy + 40 + i * 19
        screen.blit(fonts["small"].render(label, True, INK_SECONDARY), (ox + 16, y))
        value_surface = fonts["small_bold"].render(value, True, INK)
        screen.blit(value_surface, (ox + 240 - value_surface.get_width(), y))


def draw_top_bar(pygame, screen, fonts, env, scenario, speed, paused):
    width = screen.get_width()
    pygame.draw.rect(screen, SURFACE, (0, 0, width, TOP_H))
    minutes, seconds = divmod(env.t, 60)
    total_min = env.episode_length // 60
    status = "PAUSED" if paused else f"{speed:g}x speed"
    text = (f"{config.SCENARIO_LABELS[scenario]} traffic   {minutes:02d}:{seconds:02d} / "
            f"{total_min:02d}:00   {status}")
    title = fonts["bold"].render(text, True, INK)
    screen.blit(title, (16, 8))
    keys = "SPACE pause   UP/DOWN speed   R restart   S screenshot   ESC quit"
    hint = fonts["small"].render(keys, True, MUTED)
    if title.get_width() + hint.get_width() + 56 <= width:  # only when there is room
        screen.blit(hint, (width - hint.get_width() - 16, 12))
    progress = env.t / env.episode_length
    pygame.draw.rect(screen, AXIS, (16, TOP_H - 8, width - 32, 3), border_radius=2)
    pygame.draw.rect(screen, INK_SECONDARY, (16, TOP_H - 8, int((width - 32) * progress), 3),
                     border_radius=2)


def draw(pygame, screen, fonts, envs, names, scenario, speed, paused):
    screen.fill(SURFACE)
    draw_top_bar(pygame, screen, fonts, envs[0], scenario, speed, paused)
    for i, (env, name) in enumerate(zip(envs, names)):
        ox = i * PANEL_W
        draw_intersection(pygame, screen, fonts, ox, TOP_H, env)
        draw_info(pygame, screen, fonts, ox, TOP_H + PANEL_H, env, name)
        if i:
            pygame.draw.line(screen, AXIS, (ox, TOP_H), (ox, TOP_H + PANEL_H + INFO_H), 2)
    if envs[0].t >= envs[0].episode_length:
        banner = fonts["bold"].render("Episode finished - press R to restart", True, SURFACE)
        box = banner.get_rect(center=(screen.get_width() // 2, TOP_H + PANEL_H // 2)).inflate(24, 14)
        pygame.draw.rect(screen, HOUSING, box, border_radius=8)
        screen.blit(banner, banner.get_rect(center=box.center))


class Simulation:
    """Advances several environments in lock-step, one simulated second at a time."""

    def __init__(self, names, scenario, model_seed, traffic_seed):
        self.agents = [load_trained_agent(name, model_seed) for name in names]
        self.envs = [TrafficIntersectionEnv(scenario=scenario, **config.ENV_CONFIG) for _ in names]
        self.traffic_seed = traffic_seed
        self.restart()

    def restart(self):
        self.obs = [env.reset(seed=self.traffic_seed)[0] for env in self.envs]
        self.tick_in_interval = 0

    @property
    def finished(self):
        return self.envs[0].t >= self.envs[0].episode_length

    def tick(self):
        """Same order of events as TrafficIntersectionEnv.step, but one second at a time."""
        if self.finished:
            return
        if self.tick_in_interval == 0:
            for env, agent, obs in zip(self.envs, self.agents, self.obs):
                env.apply_action(agent.act(obs, greedy=True))
        for env in self.envs:
            env.tick()
        self.tick_in_interval += 1
        if self.tick_in_interval == self.envs[0].decision_interval or self.finished:
            self.obs = [env.get_obs() for env in self.envs]
            self.tick_in_interval = 0


def save_screenshot(pygame, screen, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pygame.image.save(screen, path)
    print(f"Screenshot saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=["fixed_time", "dueling_ddqn"],
                        choices=config.ALL_AGENTS)
    parser.add_argument("--scenario", default="rush_hour", choices=config.EVAL_SCENARIOS + ["random"])
    parser.add_argument("--speed", type=float, default=20, help="simulated seconds per real second")
    parser.add_argument("--model-seed", type=int, default=0, help="which trained seed to load")
    parser.add_argument("--traffic-seed", type=int, default=config.EVAL_SEED_OFFSET)
    parser.add_argument("--headless", action="store_true",
                        help="no window: simulate up to --until seconds and save --screenshot")
    parser.add_argument("--until", type=int, default=600, help="headless: simulated seconds to run")
    parser.add_argument("--screenshot", default=None, help="headless: where to save the image")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError:
        sys.exit("pygame is not installed - run: pip install pygame")

    try:
        sim = Simulation(args.agents, args.scenario, args.model_seed, args.traffic_seed)
    except FileNotFoundError as error:
        sys.exit(str(error))

    pygame.init()
    screen = pygame.display.set_mode((PANEL_W * len(args.agents), TOP_H + PANEL_H + INFO_H))
    pygame.display.set_caption("RL traffic signal control")
    fonts = {
        "bold": pygame.font.SysFont("segoeui,arial,helvetica", 17, bold=True),
        "small": pygame.font.SysFont("segoeui,arial,helvetica", 14),
        "small_bold": pygame.font.SysFont("segoeui,arial,helvetica", 14, bold=True),
    }

    if args.headless:
        while sim.envs[0].t < min(args.until, sim.envs[0].episode_length):
            sim.tick()
        draw(pygame, screen, fonts, sim.envs, args.agents, args.scenario, args.speed, False)
        path = args.screenshot or os.path.join(
            config.SCREENSHOTS_DIR, f"demo_{args.scenario}_{sim.envs[0].t}s.png")
        save_screenshot(pygame, screen, path)
        pygame.quit()
        return

    clock, speed, paused, carry = pygame.time.Clock(), args.speed, False, 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2, 480)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 2, 1)
                elif event.key == pygame.K_r:
                    sim.restart()
                elif event.key == pygame.K_s:
                    save_screenshot(pygame, screen, os.path.join(
                        config.SCREENSHOTS_DIR, f"demo_{args.scenario}_{sim.envs[0].t}s.png"))
        if not paused:
            carry += speed / FPS
            ticks, carry = int(carry), carry - int(carry)
            for _ in range(ticks):
                sim.tick()
        draw(pygame, screen, fonts, sim.envs, args.agents, args.scenario, speed, paused)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()
