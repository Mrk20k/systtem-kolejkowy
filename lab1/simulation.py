
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import random
from collections import defaultdict

# --- ---

RUN_WITH_ANIMATION = False

WORLD_SETTINGS = {
    'width': 1000,
    'height': 1000
}
BATTLEFIELD_UNITS = [
    {"id": "VSHORAD_1", "type": "C-RAM", "position": (400, 500)},
    {"id": "VSHORAD_2", "type": "C-RAM", "position": (400, 700)},
    {"id": "Wisla IBCS", "type": "LASER", "position": (600, 600)}
]
SYSTEM_TYPES = {
    "C-RAM": {
        "range": 250, "channels": 1, "service_time": 0.5,
        "damage_per_hit": 1, "magazine_size": 150, "reload_time": 600.0,
        "color": 'blue'
    },
    "LASER": {
        "range": 400, "channels": 2, "service_time": 3.0,
        "damage_per_hit": 3, "magazine_size": 12, "reload_time": 10.0,
        "color": 'red'
    }
}
DRONE_TYPES = {
    "Shahed-131": {
        "speed": 60, "hp": 1, "sway_strength": 20.0, "color": 'gray'
    },
    "Shahed-136": {
        "speed": 35, "hp": 3, "sway_strength": 10.0, "color": 'black'
    }
}
ROUTES = {
    "Route_North": { "start": (500, 1000), "target": (500, 0) },
    "Route_West": { "start": (250, 1000), "target": (500, 0) },
    "Route_East": { "start": (750, 1000), "target": (500, 0) }
}
SPAWN_WAVE = [
    {"route": "Route_North", "type": "Shahed-136", "count": 20, "interval": 0},
    {"route": "Route_West", "type": "Shahed-131", "count": 10, "interval": 0},
    {"route": "Route_East", "type": "Shahed-131", "count": 10, "interval": 0},
    {"route": "Route_North", "type": "Shahed-136", "count": 10, "interval": 0}
]
SIMULATION_SETTINGS = {
    "dt": 0.1,
    "targeting_strategy": 'NEAREST',
    "is_coordinated": True
}

def print_log(sim_time, message):
    if RUN_WITH_ANIMATION:
        print(f"[{sim_time:6.1f}s] {message}")

class Drone:
    def __init__(self, route_name, route_info, drone_type_name):
        self.x, self.y = route_info['start']
        self.target_x, self.target_y = route_info['target']
        self.route_name = route_name
        self.type_name = drone_type_name
        self.props = DRONE_TYPES[drone_type_name]
        self.speed = self.props['speed']
        self.color = self.props['color']
        self.hp = self.props['hp']
        self.sway_strength = self.props['sway_strength']
        self.status = 'active'
        self.time_in_world = 0.0
        self.vx = 0.0
        self.vy = 0.0
    def move(self, dt):
        if self.status != 'active': return
        dx_target = self.target_x - self.x
        dy_target = self.target_y - self.y
        dist_to_target = np.hypot(dx_target, dy_target)
        if self.check_status(dist_to_target, dt): return
        vx_base = (dx_target / dist_to_target) * self.speed
        vy_base = (dy_target / dist_to_target) * self.speed
        if self.speed > 0:
            perp_vx_norm = -vy_base / self.speed
            perp_vy_norm = vx_base / self.speed
        else:
            perp_vx_norm, perp_vy_norm = 0, 0
        sway_force = self.sway_strength * random.uniform(-1, 1)
        self.vx = vx_base + perp_vx_norm * sway_force
        self.vy = vy_base + perp_vy_norm * sway_force
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.time_in_world += dt
    def check_status(self, dist_to_target, dt):
        if dist_to_target < (self.speed * dt * 1.5):
            self.status = 'leaked'
            return True
        return False
    def take_hit(self, damage, sim_time):
        if self.status != 'active': return
        self.hp -= damage
        if self.hp <= 0:
            self.status = 'destroyed'
            print_log(sim_time, f"Dron {self.type_name} zniszczony!")
    def get_distance_to_goal(self):
        return np.hypot(self.target_x - self.x, self.target_y - self.y)

class DefenseSystem:
    """Reprezentuje system obronny (serwer). Zarządza swoim stanem."""
    def __init__(self, unit_info, system_types_dict):
        self.id = unit_info['id']
        self.x, self.y = unit_info['position']
        self.type = unit_info['type']
        self.props = system_types_dict[self.type]
        self.range = self.props['range']
        self.channels = self.props['channels']
        self.service_time = self.props['service_time']
        self.damage = self.props['damage_per_hit']
        self.magazine_size = self.props['magazine_size']
        self.reload_time = self.props['reload_time']
        self.color = self.props['color']
        self.status = 'IDLE'
        self.current_ammo = self.magazine_size
        self.reload_timer = 0.0
        self.targets = {}
        self.total_time_busy = 0.0 #
        self.total_time_reloading = 0.0
        self.targets_destroyed = 0
        
        self.total_channel_busy_time = 0.0

    def get_free_channels(self):
        return self.channels - len(self.targets)
    def can_engage(self):
        return self.status != 'RELOADING'
    def is_targeting(self, drone):
        return drone in self.targets

    def update_state(self, dt, sim_time):
        """Aktualizuje wewnętrzne timery (przeładowanie, obsługa celu)."""
        
        if self.status == 'RELOADING':
            self.reload_timer -= dt
            self.total_time_reloading += dt
            if self.reload_timer <= 0:
                self.status = 'IDLE'
                self.current_ammo = self.magazine_size
                print_log(sim_time, f"System {self.id} zakończył przeładowanie")
            return
        
        num_busy_channels = len(self.targets)
        self.total_channel_busy_time += num_busy_channels * dt

        if not self.targets:
            self.status = 'IDLE'
            return

        self.status = 'BUSY'
        self.total_time_busy += dt
            
        for drone, timer in list(self.targets.items()):
            dist = np.hypot(drone.x - self.x, drone.y - self.y)
            if drone.status != 'active' or dist > self.range:
                self._remove_target(drone)
                continue 
            
            timer -= dt
            
            if timer <= 0:
                if self.current_ammo > 0:
                    self._fire_at_target(drone, sim_time)
                    if drone.status == 'destroyed':
                        self._remove_target(drone)
                        self.targets_destroyed += 1
                    else:
                        self.targets[drone] = self.service_time 
                else:
                    print_log(sim_time, f"System {self.id} Brak amunicji!")
                    self._start_reloading(sim_time)
                    break
            else:
                self.targets[drone] = timer 

    def engage_target(self, drone, sim_time):
        if not self.can_engage() or self.get_free_channels() <= 0:
            return
        self.targets[drone] = self.service_time
        print_log(sim_time, f"System {self.id} namierzył {drone.type_name}")

    def _fire_at_target(self, drone, sim_time):
        if self.current_ammo <= 0: return
        self.current_ammo -= 1
        print_log(sim_time, f"System {self.id} strzela do {drone.type_name}! (Amunicja: {self.current_ammo})")
        drone.take_hit(self.damage, sim_time)
        if self.current_ammo <= 0 and self.magazine_size > 0:
            print_log(sim_time, f"System {self.id} [Ostatni Strzał]... ")
            self._start_reloading(sim_time)

    def _start_reloading(self, sim_time):
        if self.status == 'RELOADING': return
        print_log(sim_time, f"System {self.id} przeładowuje ({self.reload_time}s)")
        self.status = 'RELOADING'
        self.reload_timer = self.reload_time
        self.targets.clear() 

    def _remove_target(self, drone):
        if drone in self.targets:
            del self.targets[drone]

    def get_distance_to_drone(self, drone):
        return np.hypot(drone.x - self.x, drone.y - self.y)

class Simulation:
    def __init__(self, world, sim_settings, system_types, units, drone_types, routes, wave):
        self.world = world
        self.sim_settings = sim_settings
        self.system_types = system_types
        self.drone_types = drone_types
        self.routes = routes
        self.sim_time = 0.0
        self.dt = sim_settings['dt']
        self.active_drones = []
        self.leaked_drones = []
        self.destroyed_drones = []
        self.systems = [DefenseSystem(u, self.system_types) for u in units]
        self.spawn_plan = list(wave)
        self.next_spawn_time = 0.0
        self.spawn_interval = 0.0
        self.drones_to_spawn_in_batch = 0
        self.current_batch_info = {}
        self._setup_next_batch()
    def _setup_next_batch(self):
        if not self.spawn_plan:
            self.drones_to_spawn_in_batch = 0
            print_log(self.sim_time, "Wszystkie fale dronów zostały wysłane.")
            return
        self.current_batch_info = self.spawn_plan.pop(0)
        self.drones_to_spawn_in_batch = self.current_batch_info['count']
        self.spawn_interval = self.current_batch_info['interval']
        self.next_spawn_time = self.sim_time + self.spawn_interval
        msg = f"FALA: {self.current_batch_info['type']} na {self.current_batch_info['route']}"
        print_log(self.sim_time, msg)
    def _spawn_drone(self):
        route_name = self.current_batch_info['route']
        route_info = self.routes[route_name]
        drone_type = self.current_batch_info['type']
        new_drone = Drone(route_name, route_info, drone_type)
        self.active_drones.append(new_drone)
        self.drones_to_spawn_in_batch -= 1
        self.next_spawn_time += self.spawn_interval
        if self.drones_to_spawn_in_batch <= 0:
            self._setup_next_batch()
    def step(self):
        if self.drones_to_spawn_in_batch > 0 and self.sim_time >= self.next_spawn_time:
            self._spawn_drone()
        for drone in self.active_drones[:]:
            drone.move(self.dt)
            if drone.status == 'leaked':
                self.active_drones.remove(drone)
                self.leaked_drones.append(drone)
                print_log(self.sim_time, f"PRZECIEK: {drone.type_name} na {drone.route_name}.")
            elif drone.status == 'destroyed':
                self.active_drones.remove(drone)
                self.destroyed_drones.append(drone)
        for system in self.systems:
            system.update_state(self.dt, self.sim_time)
        self._assign_new_targets()
        self.sim_time += self.dt
    def _assign_new_targets(self):
        globally_engaged_drones = set()
        if self.sim_settings['is_coordinated']:
            for s in self.systems:
                globally_engaged_drones.update(s.targets.keys())
        for system in self.systems:
            free_channels = system.get_free_channels()
            if not system.can_engage() or free_channels <= 0:
                continue
            if system.magazine_size > 0:
                if system.current_ammo <= len(system.targets):
                    continue
            potential_targets = []
            for drone in self.active_drones:
                dist = system.get_distance_to_drone(drone)
                if dist > system.range: continue
                if system.is_targeting(drone): continue
                if self.sim_settings['is_coordinated'] and drone in globally_engaged_drones:
                    continue
                potential_targets.append(drone)
            if not potential_targets: continue
            strategy = self.sim_settings['targeting_strategy']
            if strategy == 'NEAREST':
                potential_targets.sort(key=lambda d: system.get_distance_to_drone(d))
            elif strategy == 'OLDEST':
                potential_targets.sort(key=lambda d: d.time_in_world, reverse=True)
            elif strategy == 'NEAREST_GOAL':
                potential_targets.sort(key=lambda d: d.get_distance_to_goal())
            targets_to_assign = potential_targets[:free_channels]
            for target in targets_to_assign:
                system.engage_target(target, self.sim_time)
                if self.sim_settings['is_coordinated']:
                    globally_engaged_drones.add(target)
    def is_finished(self):
        no_active = not self.active_drones
        no_more_spawns = self.drones_to_spawn_in_batch <= 0 and not self.spawn_plan
        return no_active and no_more_spawns

fig, ax = plt.subplots(figsize=(10, 10))
ani = None
def draw_static_environment(ax, sim_instance):
    ax.set_aspect('equal')
    ax.set_xlim(0, sim_instance.world['width'])
    ax.set_ylim(0, sim_instance.world['height'])
    ax.grid(True, linestyle=':', alpha=0.6)
    legend_handles = []
    for route_name, route_info in sim_instance.routes.items():
        start, target = route_info['start'], route_info['target']
        ax.plot([start[0], target[0]], [start[1], target[1]], 'k:', alpha=0.3)
        h, = ax.plot([], [], 'X', color='black', markersize=15, alpha=0.5, label=f"Cel: {route_name}")
        legend_handles.append(h)
        ax.plot(target[0], target[1], 'X', color='black', markersize=15, alpha=0.5)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, 0.95))
def draw_systems(ax, sim_instance):
    for system in sim_instance.systems:
        x, y = system.x, system.y
        status_color = 'gray'
        if system.status == 'BUSY': status_color = 'red'
        elif system.status == 'RELOADING': status_color = 'orange'
        zone = patches.Circle((x, y), system.range, facecolor=status_color, alpha=0.1, edgecolor=status_color, linestyle='--')
        ax.add_patch(zone)
        ax.plot(x, y, 'x', color=system.color, markersize=10, markeredgewidth=3)
        ammo_text = f"{system.current_ammo}" if system.magazine_size > 0 else "Inf"
        if system.status == 'RELOADING':
            ammo_text = f"RELOAD ({system.reload_timer:.1f}s)"
        ax.text(x + 10, y + 10, f"{system.id}\nAMMO: {ammo_text}", fontsize=9, color='black', ha='left')
        for drone in system.targets:
            ax.plot([x, drone.x], [y, drone.y], color='red', linestyle='-', linewidth=0.5)
def draw_drones(ax, sim_instance):
    for drone in sim_instance.active_drones:
        ax.plot(drone.x, drone.y, 'o', color=drone.color, markersize=6)
        if drone.hp < DRONE_TYPES[drone.type_name]['hp']:
             ax.text(drone.x, drone.y - 10, f"HP: {drone.hp}", fontsize=8, color='red', ha='center')
    for drone in sim_instance.destroyed_drones:
        ax.plot(drone.x, drone.y, 'x', color='red', markersize=5, alpha=0.7)
    for drone in sim_instance.leaked_drones:
        ax.plot(drone.target_x, drone.target_y, 'v', color='gray', markersize=8, alpha=0.7)
def update_plot(frame, sim_instance):
    global ani
    if not sim_instance.is_finished():
        sim_instance.step()
    ax.clear()
    draw_static_environment(ax, sim_instance)
    draw_systems(ax, sim_instance)
    draw_drones(ax, sim_instance)
    ax.set_title(f"Symulacja | Czas: {sim_instance.sim_time:.1f}s | "
                 f"Zniszczone: {len(sim_instance.destroyed_drones)} | "
                 f"Przeciekło: {len(sim_instance.leaked_drones)}")
    if sim_instance.is_finished():
        if ani: ani.event_source.stop()
        plot_final_statistics(sim_instance) 

# --- Krok 5.1: NOWY DASHBOARD STATYSTYK ---

def plot_final_statistics(sim_instance):

    num_destroyed = len(sim_instance.destroyed_drones)
    num_leaked = len(sim_instance.leaked_drones)
    total_drones = num_destroyed + num_leaked
    sim_time = sim_instance.sim_time
    
    if total_drones == 0:
        print("Brak dronów do analizy.")
        return
    if sim_time == 0:
        print("Czas symulacji = 0. Nie można wygenerować statystyk.")
        return

    # Inicjalizacja Dashboardu
    fig_stats, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig_stats.suptitle(f"Raport Końcowy (Strategia: {sim_instance.sim_settings['targeting_strategy']}, Koor: {sim_instance.sim_settings['is_coordinated']})", fontsize=16)

    # --- Wykres 1: Skuteczność Ogólna (Kołowy) ---
    ax1 = axs[0, 0]
    sizes = [num_destroyed, num_leaked]
    labels = [f'Zniszczone ({num_destroyed})', f'Przeciek ({num_leaked})']
    colors = ['#86C166', '#DB3A2C']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white'})
    ax1.set_title('Skuteczność Ogólna')
    ax1.axis('equal')

    # --- Wykres 2: Nieszczelność Tras (Przeniesiony) ---
    ax2 = axs[0, 1]
    leak_by_route = defaultdict(int)
    for drone in sim_instance.leaked_drones:
        leak_by_route[drone.route_name] += 1
    
    if not leak_by_route:
        ax2.text(0.5, 0.5, "BRAK PRZECIEKÓW!\n100% Skuteczności!", ha='center', va='center', fontsize=18, color='green')
        ax2.set_title('Nieszczelność Tras')
    else:
        routes = list(leak_by_route.keys())
        leaks = list(leak_by_route.values())
        ax2.bar(routes, leaks, color='gray')
        ax2.set_ylabel('Liczba dronów, które przeciekły')
        ax2.set_title('Nieszczelność Tras')
        for i, v in enumerate(leaks):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

    # --- Wykres 3: Efektywność Systemów (Kto zniszczył ile) ---
    ax3 = axs[1, 0]
    system_names = [s.id for s in sim_instance.systems]
    destroyed_counts = [s.targets_destroyed for s in sim_instance.systems]
    ax3.bar(system_names, destroyed_counts, color=['blue', 'blue', 'red'])
    ax3.set_ylabel('Liczba zniszczonych dronów')
    ax3.set_title('Efektywność Systemów')
    for i, v in enumerate(destroyed_counts):
        ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        
    # --- Wykres 4: NOWA TABELA (Statystyki Kolejkowe) ---
    ax4 = axs[1, 1]
    ax4.axis('off') 
    ax4.set_title('Statystyki Systemów (Model Kolejkowy)')

    cell_data = []
    row_labels = []
    for s in sim_instance.systems:
        row_labels.append(s.id)
        
        # 1. Średnia liczba zajętych kanałów
        avg_channels = s.total_channel_busy_time / sim_time
        
        # 2. Utylizacja kanałów (%)
        total_available_channel_time = s.channels * sim_time
        util_percent = (s.total_channel_busy_time / total_available_channel_time) * 100
        
        # 3. Średni czas systemu na zniszczenie
        if s.targets_destroyed > 0:
            avg_time_per_kill = s.total_time_busy / s.targets_destroyed
        else:
            avg_time_per_kill = 0.0 

        cell_data.append([
            f"{avg_channels:.2f}",
            f"{util_percent:.1f} %",
            f"{avg_time_per_kill:.2f} s"
        ])

    col_labels = ["Śr. zajęte kanały", "Utylizacja kanałów (%)", "Śr. czas / zniszczenie (s)"]
    table = ax4.table(cellText=cell_data, colLabels=col_labels, rowLabels=row_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_headless_simulation():
    start_time = time.time()
    
    sim = Simulation(WORLD_SETTINGS, SIMULATION_SETTINGS, SYSTEM_TYPES, 
                     BATTLEFIELD_UNITS, DRONE_TYPES, ROUTES, SPAWN_WAVE)
    
    while not sim.is_finished():
        sim.step()
        
    end_time = time.time()
    print(f"--- Symulacja Zakończona ---")
    print(f"Czas obliczeń: {end_time - start_time:.2f}s")
    print(f"Czas symulacji: {sim.sim_time:.1f}s")
    print(f"Zniszczone: {len(sim.destroyed_drones)}")
    print(f"Przeciekło: {len(sim.leaked_drones)}")
    
    plot_final_statistics(sim)

def run_animated_simulation():
    global ani, sim
    
    sim = Simulation(WORLD_SETTINGS, SIMULATION_SETTINGS, SYSTEM_TYPES, 
                     BATTLEFIELD_UNITS, DRONE_TYPES, ROUTES, SPAWN_WAVE)
    
    ani = animation.FuncAnimation(fig, update_plot, fargs=(sim,),
                                  interval=(sim.dt * 1000), 
                                  blit=False)
    plt.show()

if __name__ == "__main__":
    if RUN_WITH_ANIMATION:
        run_animated_simulation()
    else:
        plt.close(fig) 
        run_headless_simulation()