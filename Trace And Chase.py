import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider, RadioButtons, CheckButtons
import time

# ============================================================================
# GUIDANCE ALGORITHM SELECTION
# ============================================================================
# Available algorithms:
# - "pure_pursuit"    : Missile heads directly toward target's current position
# - "lead_pursuit"    : Missile predicts where target will be and aims ahead
# - "proportional_navigation" : Uses line-of-sight rate (realistic missile guidance)
# - "augmented_pn"    : Proportional Navigation with target acceleration compensation

GUIDANCE_ALGORITHM = "proportional_navigation"  # Change this to select algorithm
PN_CONSTANT = 4  # Navigation constant for PN (typically 3-5)

# ============================================================================
# PHYSICS SETTINGS
# ============================================================================
# Enable/disable physics effects
ENABLE_PHYSICS = True  # Master switch for all physics effects

# Gravity
ENABLE_GRAVITY = True
GRAVITY = 9.81  # m/s^2 (Earth's gravity)

# Aerodynamic Drag
ENABLE_DRAG = True
# Drag coefficient: F_drag = 0.5 * rho * v^2 * Cd * A
# Simplified: a_drag = DRAG_COEFFICIENT * v^2 (combined constant)
DRAG_COEFFICIENT = 0.00001  # Adjust for desired drag effect

# Turn Rate Limit (maximum angular velocity the missile can turn)
ENABLE_TURN_LIMIT = True
MAX_TURN_RATE = 20.0  # degrees per second (typical missile: 15-30 deg/s)
MAX_TURN_RATE_RAD = np.radians(MAX_TURN_RATE)  # Convert to radians

# Thrust and Fuel
ENABLE_FUEL = True
FUEL_DURATION = 30.0  # seconds of powered flight
THRUST_ACCELERATION = 50.0  # m/s^2 acceleration from thrust (when fuel available)

# Missile minimum speed (below this, missile is considered "dead")
MIN_MISSILE_SPEED = 100.0  # m/s

# ============================================================================
# EVASIVE MANEUVERS SETTINGS
# ============================================================================
# Enable evasive maneuvers for the target aircraft
ENABLE_EVASION = True

# Available evasion patterns:
# - "none"          : No evasion (original trajectory)
# - "jinking"       : Random direction changes (zigzag)
# - "barrel_roll"   : Spiral/corkscrew motion
# - "break_turn"    : Hard turn when missile gets close
# - "weave"         : Sinusoidal weaving pattern
# - "split_s"       : Dive and reverse direction
# - "random"        : Randomly switch between patterns

EVASION_PATTERN = "jinking"  # Select evasion pattern

# Evasion parameters
EVASION_START_DISTANCE = 5000.0  # Start evading when missile is this close (m)
EVASION_INTENSITY = 1.0  # Multiplier for evasion magnitude (0.5 = mild, 2.0 = aggressive)

# Jinking parameters
JINK_INTERVAL = 2.0  # seconds between direction changes
JINK_AMPLITUDE = 500.0  # maximum lateral displacement (m)

# Barrel roll parameters
BARREL_ROLL_RADIUS = 300.0  # radius of the roll (m)
BARREL_ROLL_RATE = 0.5  # rolls per second

# Break turn parameters
BREAK_TURN_RATE = 8.0  # degrees per second
BREAK_TURN_DISTANCE = 3000.0  # trigger distance for break turn

# Weave parameters
WEAVE_FREQUENCY = 0.3  # Hz
WEAVE_AMPLITUDE = 400.0  # meters

# ============================================================================
# MULTI-MISSILE SETTINGS (Heat-Seeker Configuration)
# ============================================================================
# Enable multiple missiles
ENABLE_MULTI_MISSILE = True
NUM_MISSILES = 3  # Number of missiles to launch

# Sequential launch mode - launch next missile only if previous fails
SEQUENTIAL_LAUNCH = True  # Launch one at a time, next only if previous fails

# Missile launch configuration - Heat-Seeker missiles from different launchers
# Heat-seekers use pure_pursuit (tracks heat signature directly)
MISSILE_CONFIGS = [
    {
        "start_pos": np.array([20000, 0, 500]),      # Launcher 1 - East
        "launch_time": 0.0,
        "guidance": "pure_pursuit",  # Heat-seeker
        "color": "red",
        "name": "SAM-1"
    },
    {
        "start_pos": np.array([0, 20000, 800]),      # Launcher 2 - North
        "launch_time": 0.0,  # Will be set dynamically if SEQUENTIAL_LAUNCH
        "guidance": "pure_pursuit",  # Heat-seeker
        "color": "orange",
        "name": "SAM-2"
    },
    {
        "start_pos": np.array([-15000, 10000, 300]), # Launcher 3 - Northwest
        "launch_time": 0.0,  # Will be set dynamically if SEQUENTIAL_LAUNCH
        "guidance": "pure_pursuit",  # Heat-seeker
        "color": "magenta",
        "name": "SAM-3"
    },
]

# Salvo mode: launch all missiles at intervals
SALVO_MODE = False
SALVO_INTERVAL = 1.0  # seconds between each launch in salvo

# Target assignment for multiple missiles
# "single" - all missiles target the same aircraft
# "distributed" - missiles are assigned to different targets (if multiple targets exist)
TARGET_ASSIGNMENT = "single"

# ============================================================================
# AUTO-RESTART SETTINGS
# ============================================================================
AUTO_RESTART = True  # Automatically restart simulation after intercept
RESTART_DELAY = 10.0  # Seconds to wait before restart (pause when Jet destroyed)

# ============================================================================
# COUNTERMEASURES SETTINGS (Heat-Seeker Defense)
# ============================================================================
# Enable countermeasures system
ENABLE_COUNTERMEASURES = True

# Countermeasure types:
# - "flare"   : IR decoy - attracts heat-seeking missiles
# - "chaff"   : Radar decoy - NOT effective against heat-seekers
# - "ecm"     : Electronic countermeasures - NOT effective against heat-seekers

# Flare settings (PRIMARY defense against heat-seekers)
ENABLE_FLARES = True
FLARE_COUNT = 2  # Aircraft has only 2 flares!
FLARE_DEPLOY_DISTANCE = 1500.0  # Deploy when missile is VERY close (m) - risky but realistic
FLARE_DEPLOY_INTERVAL = 3.0  # Minimum time between flare deployments (s)
FLARE_EFFECTIVENESS = 0.7  # 70% chance of successfully decoying heat-seeker
FLARE_DURATION = 5.0  # How long flare remains effective (s)
FLARE_FALL_RATE = 50.0  # How fast flare falls (m/s)

# Chaff settings - DISABLED for heat-seeker scenario
ENABLE_CHAFF = False
CHAFF_COUNT = 0
CHAFF_DEPLOY_DISTANCE = 4000.0
CHAFF_DEPLOY_INTERVAL = 3.0
CHAFF_EFFECTIVENESS = 0.0  # Not effective against heat-seekers
CHAFF_CLOUD_RADIUS = 200.0
CHAFF_DURATION = 8.0

# ECM settings - DISABLED for heat-seeker scenario
ENABLE_ECM = False
ECM_RANGE = 5000.0
ECM_EFFECTIVENESS = 0.0  # Not effective against heat-seekers
ECM_GUIDANCE_NOISE = 0.0

# Countermeasure susceptibility by missile guidance type
# Heat-seekers (pure_pursuit) are highly susceptible to flares
CM_SUSCEPTIBILITY = {
    "pure_pursuit": {"flare": 0.9, "chaff": 0.0, "ecm": 0.0},  # Heat-seeker
    "lead_pursuit": {"flare": 0.7, "chaff": 0.3, "ecm": 0.4},
    "proportional_navigation": {"flare": 0.5, "chaff": 0.5, "ecm": 0.5},
    "augmented_pn": {"flare": 0.4, "chaff": 0.4, "ecm": 0.6},
}

# ============================================================================
# MONTE CARLO ANALYSIS SETTINGS
# ============================================================================
# Enable Monte Carlo mode (runs multiple simulations for statistical analysis)
ENABLE_MONTE_CARLO = False  # Set to True to run Monte Carlo analysis

# Number of simulation runs
MONTE_CARLO_RUNS = 100

# Parameter randomization settings
MC_RANDOMIZE_MISSILE_POS = True  # Randomize missile starting position
MC_MISSILE_POS_VARIANCE = 2000.0  # Standard deviation in meters

MC_RANDOMIZE_TARGET_POS = True  # Randomize target starting position
MC_TARGET_POS_VARIANCE = 500.0  # Standard deviation in meters

MC_RANDOMIZE_VELOCITIES = True  # Randomize initial velocities
MC_VELOCITY_VARIANCE = 50.0  # Standard deviation in m/s

MC_RANDOMIZE_EVASION = True  # Randomize evasion parameters
MC_EVASION_PATTERNS = ["none", "jinking", "weave", "barrel_roll", "break_turn"]

MC_RANDOMIZE_GUIDANCE = True  # Test different guidance algorithms
MC_GUIDANCE_ALGORITHMS = ["pure_pursuit", "lead_pursuit", "proportional_navigation", "augmented_pn"]

# Statistical output options
MC_SHOW_HISTOGRAMS = True
MC_SHOW_SCATTER = True
MC_SAVE_RESULTS = False  # Save results to CSV file
MC_RESULTS_FILE = "monte_carlo_results.csv"

# ============================================================================
# INTERACTIVE GUI SETTINGS
# ============================================================================
# Enable interactive control panel
ENABLE_INTERACTIVE_GUI = False  # Disabled - using custom status panel instead

# GUI layout options
GUI_PANEL_WIDTH = 0.25  # Width of control panel (0-1)
GUI_SHOW_SLIDERS = True  # Show velocity sliders
GUI_SHOW_BUTTONS = True  # Show play/pause/reset buttons
GUI_SHOW_DROPDOWNS = True  # Show algorithm/evasion selectors

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Enhanced visualization options
SHOW_VELOCITY_VECTORS = True  # Show direction arrows on aircraft and missiles
VELOCITY_VECTOR_SCALE = 0.1  # Scale factor for velocity vectors

SHOW_DISTANCE_LINES = True  # Show lines between missiles and target
DISTANCE_LINE_COLOR = 'gray'
DISTANCE_LINE_ALPHA = 0.3

SHOW_GROUND_PLANE = True  # Show ground reference plane
GROUND_PLANE_ALPHA = 0.1
GROUND_PLANE_COLOR = 'green'

SHOW_ALTITUDE_LINES = True  # Show vertical lines to ground
ALTITUDE_LINE_ALPHA = 0.2

TRAIL_FADE_EFFECT = True  # Trails fade over time
TRAIL_MAX_LENGTH = 500  # Maximum trail length in frames

SHOW_INTERCEPT_PREDICTION = True  # Show predicted intercept point

# ============================================================================
# Variables
# ============================================================================
Straight_time = 25      # Duration of first straight segment (s)
curve_time = 25         # Duration of turn (s)
Straight_time2 = 25     # Duration of second straight segment (s)
targ_vel = 750          # Target velocity (m/s)
miss_vel = 800          # missile velocity (m/s)
turn_angle = -np.pi*4/3   # Turn angle in radians (np.pi = 180°, np.pi/2 = 90°, etc.)
tmax = 75
dt = 0.001
animation_interval = 5  # milliseconds
yz_angle = -np.pi/12
missile_start_loc = np.array([13000, 12000, 0])
aircraft_start_loc = np.array([0, 0, 12000])  # Aircraft starting position
missile_launch_time = 0  # Time when missile launches (s)
kill_dist=2
climb_rate_curve= -0.001

# ============================================================================
# RANDOMIZATION SETTINGS
# ============================================================================
# Randomize aircraft starting position
RANDOMIZE_AIRCRAFT_START = True
AIRCRAFT_X_RANGE = (-2000, 2000)  # Random X offset range
AIRCRAFT_Y_RANGE = (-2000, 2000)  # Random Y offset range
AIRCRAFT_Z_RANGE = (8000, 15000)  # Random altitude range

# Apply randomization to aircraft start position
if RANDOMIZE_AIRCRAFT_START:
    rng_start = np.random.default_rng()
    aircraft_start_loc = np.array([
        rng_start.uniform(AIRCRAFT_X_RANGE[0], AIRCRAFT_X_RANGE[1]),
        rng_start.uniform(AIRCRAFT_Y_RANGE[0], AIRCRAFT_Y_RANGE[1]),
        rng_start.uniform(AIRCRAFT_Z_RANGE[0], AIRCRAFT_Z_RANGE[1])
    ])
    print(f"Randomized aircraft start position: ({aircraft_start_loc[0]:.1f}, {aircraft_start_loc[1]:.1f}, {aircraft_start_loc[2]:.1f})")

# ============================================================================
# Global variables for segment start positions
# ============================================================================
curve_start_x = None
curve_start_y = None
curve_start_z = None
curve_initialized = False

straight2_start_x = None
straight2_start_y = None
straight2_start_z = None
straight2_initialized = False

radius = (targ_vel * curve_time) / turn_angle  # Calculate radius once
center_x = None
center_y = None
center_z = None

# ============================================================================
# TARGET LOCATION
# ============================================================================
def target_location(t, target_states):
    """
    Calculate target position at time t.

    Parameters:
    - t: time (s)
    - target_states: array of previous target states for initialization

    Returns:
    - np.array([x, y, z]): position in meters
    """
    global curve_start_x, curve_start_y, curve_start_z, curve_initialized
    global straight2_start_x, straight2_start_y, straight2_start_z, straight2_initialized
    global center_x, center_y, center_z

    if 0 <= t <= Straight_time:
        # Straight flight in +X direction from starting position
        x = aircraft_start_loc[0] + targ_vel * t
        y = aircraft_start_loc[1]
        z = aircraft_start_loc[2]

    elif Straight_time < t <= Straight_time + curve_time:
        # Curved turn with proper center
        tc = t - Straight_time
        # First time entering curve segment - store start position
        if not curve_initialized:
            if len(target_states) > 0:
                curve_start_x = target_states[-1, 0]
                curve_start_y = target_states[-1, 1]
                curve_start_z = target_states[-1, 2]
            else:
                # Fallback for edge case
                curve_start_x = aircraft_start_loc[0] + targ_vel * Straight_time
                curve_start_y = aircraft_start_loc[1]
                curve_start_z = aircraft_start_loc[2]

            # Define center of circular arc
            center_x = curve_start_x
            center_y = curve_start_y + radius * np.cos(yz_angle)
            center_z = curve_start_z + radius * np.sin(yz_angle)

            curve_initialized = True

        angle = tc * turn_angle / curve_time  # 0 to turn_angle over curve_time

        # Position on circular arc
        # Start at angle = -pi/2 (pointing in +X direction from center)
        arc_angle = -np.pi/2 + angle

        x = center_x + radius * np.cos(arc_angle)
        y = center_y + radius * np.sin(arc_angle) * np.cos(yz_angle) + np.cos(yz_angle+np.pi/2) * targ_vel**2 * (1-np.cos(np.pi*tc/curve_time)) * climb_rate_curve
        z = center_z + radius * np.sin(arc_angle) * np.sin(yz_angle) + np.sin(yz_angle+np.pi/2) * targ_vel**2 * (1-np.cos(np.pi*tc/curve_time)) * climb_rate_curve

    elif Straight_time + curve_time < t <= Straight_time + curve_time + Straight_time2:
        # Straight flight after turn

        # First time entering second straight segment - store start position
        if not straight2_initialized:
            if len(target_states) > 0:
                straight2_start_x = target_states[-1, 0]
                straight2_start_y = target_states[-1, 1]
                straight2_start_z = target_states[-1, 2]
            else:
                # Fallback for edge case
                straight2_start_x = curve_start_x
                straight2_start_y = curve_start_y
                straight2_start_z = curve_start_z

            straight2_initialized = True

        ts = t - (Straight_time + curve_time)

        # Direction after turn (rotated by turn_angle from initial +X direction)
        dx = np.cos(turn_angle)
        dy = np.sin(turn_angle) * np.cos(yz_angle)
        dz = np.sin(turn_angle) * np.sin(yz_angle)

        # Continue in new direction
        x = straight2_start_x + targ_vel * ts * dx
        y = straight2_start_y + targ_vel * ts * dy
        z = straight2_start_z + targ_vel * ts * dz

    else:
        # Out of bounds - return last known position
        if straight2_initialized:
            ts_max = Straight_time2
            dx = np.cos(turn_angle)
            dy = np.sin(turn_angle) * np.cos(yz_angle)
            dz = np.sin(turn_angle) * np.sin(yz_angle)
            x = straight2_start_x + targ_vel * ts_max * dx
            y = straight2_start_y + targ_vel * ts_max * dy
            z = straight2_start_z + targ_vel * ts_max * dz
        else:
            # Fallback if never initialized
            x = aircraft_start_loc[0] + targ_vel * Straight_time
            y = aircraft_start_loc[1]
            z = aircraft_start_loc[2]

    return np.array([x, y, z])

# ============================================================================
# GUIDANCE ALGORITHMS
# ============================================================================

def pure_pursuit_guidance(missile_pos, target_pos, target_vel, missile_vel_magnitude, dt, prev_los=None):
    """
    Pure Pursuit: Missile heads directly toward target's current position.
    Simple but not optimal - missile always "chases" the target.

    Returns: (new_position, new_los_vector)
    """
    direction = target_pos - missile_pos
    distance = np.linalg.norm(direction)

    if distance > 0:
        unit_vec = direction / distance
        new_pos = missile_pos + unit_vec * missile_vel_magnitude * dt
    else:
        new_pos = missile_pos
        unit_vec = np.array([1, 0, 0])

    return new_pos, unit_vec


def lead_pursuit_guidance(missile_pos, target_pos, target_vel, missile_vel_magnitude, dt, prev_los=None):
    """
    Lead Pursuit: Missile predicts where target will be and aims ahead.
    Estimates intercept point based on closing geometry.

    Returns: (new_position, new_los_vector)
    """
    # Calculate time to intercept (approximate)
    distance = np.linalg.norm(target_pos - missile_pos)
    closing_speed = missile_vel_magnitude  # Simplified assumption

    if closing_speed > 0:
        time_to_intercept = distance / closing_speed
    else:
        time_to_intercept = 0

    # Predict target position at intercept time
    predicted_target_pos = target_pos + target_vel * time_to_intercept

    # Aim at predicted position
    direction = predicted_target_pos - missile_pos
    dist = np.linalg.norm(direction)

    if dist > 0:
        unit_vec = direction / dist
        new_pos = missile_pos + unit_vec * missile_vel_magnitude * dt
    else:
        new_pos = missile_pos
        unit_vec = np.array([1, 0, 0])

    return new_pos, unit_vec


def proportional_navigation_guidance(missile_pos, target_pos, target_vel, missile_vel_magnitude, dt, prev_los=None):
    """
    Proportional Navigation (PN): Uses line-of-sight rate for guidance.
    The most common and effective missile guidance law.

    Command acceleration is proportional to:
    a_cmd = N * V_c * (d_lambda/dt)

    Where:
    - N = Navigation constant (typically 3-5)
    - V_c = Closing velocity
    - d_lambda/dt = Line-of-sight rate

    Returns: (new_position, new_los_vector)
    """
    # Calculate line-of-sight vector
    los = target_pos - missile_pos
    distance = np.linalg.norm(los)

    if distance < 1e-6:
        return missile_pos, prev_los if prev_los is not None else np.array([1, 0, 0])

    los_unit = los / distance

    # If no previous LOS, use pure pursuit for this step
    if prev_los is None:
        unit_vec = los_unit
        new_pos = missile_pos + unit_vec * missile_vel_magnitude * dt
        return new_pos, los_unit

    # Calculate LOS rate (angular velocity)
    los_rate = (los_unit - prev_los) / dt

    # Calculate closing velocity
    missile_vel_vec = los_unit * missile_vel_magnitude  # Approximate current missile velocity direction
    relative_vel = target_vel - missile_vel_vec
    closing_velocity = -np.dot(relative_vel, los_unit)

    # PN command: acceleration perpendicular to LOS
    # a_cmd = N * Vc * LOS_rate
    accel_cmd = PN_CONSTANT * closing_velocity * los_rate

    # Current missile velocity (assume pointing along previous LOS)
    current_vel = prev_los * missile_vel_magnitude

    # Apply acceleration to get new velocity
    new_vel = current_vel + accel_cmd * dt

    # Normalize to maintain constant speed
    new_vel_magnitude = np.linalg.norm(new_vel)
    if new_vel_magnitude > 0:
        new_vel = new_vel / new_vel_magnitude * missile_vel_magnitude

    # Update position
    new_pos = missile_pos + new_vel * dt

    # Return new LOS for next iteration
    new_los = (target_pos - new_pos)
    new_los_dist = np.linalg.norm(new_los)
    if new_los_dist > 0:
        new_los = new_los / new_los_dist

    return new_pos, new_los


def augmented_pn_guidance(missile_pos, target_pos, target_vel, missile_vel_magnitude, dt, prev_los=None, target_accel=None):
    """
    Augmented Proportional Navigation (APN): PN with target acceleration compensation.
    Better performance against maneuvering targets.

    a_cmd = N * V_c * (d_lambda/dt) + (N/2) * a_t_perp

    Where a_t_perp is target acceleration perpendicular to LOS.

    Returns: (new_position, new_los_vector)
    """
    # Calculate line-of-sight vector
    los = target_pos - missile_pos
    distance = np.linalg.norm(los)

    if distance < 1e-6:
        return missile_pos, prev_los if prev_los is not None else np.array([1, 0, 0])

    los_unit = los / distance

    # If no previous LOS, use pure pursuit for this step
    if prev_los is None:
        unit_vec = los_unit
        new_pos = missile_pos + unit_vec * missile_vel_magnitude * dt
        return new_pos, los_unit

    # Calculate LOS rate
    los_rate = (los_unit - prev_los) / dt

    # Calculate closing velocity
    missile_vel_vec = los_unit * missile_vel_magnitude
    relative_vel = target_vel - missile_vel_vec
    closing_velocity = -np.dot(relative_vel, los_unit)

    # Standard PN term
    accel_pn = PN_CONSTANT * closing_velocity * los_rate

    # Augmentation term for target acceleration
    accel_aug = np.zeros(3)
    if target_accel is not None:
        # Get target acceleration perpendicular to LOS
        accel_parallel = np.dot(target_accel, los_unit) * los_unit
        accel_perp = target_accel - accel_parallel
        accel_aug = (PN_CONSTANT / 2) * accel_perp

    # Total commanded acceleration
    accel_cmd = accel_pn + accel_aug

    # Current missile velocity
    current_vel = prev_los * missile_vel_magnitude

    # Apply acceleration
    new_vel = current_vel + accel_cmd * dt

    # Normalize to maintain constant speed
    new_vel_magnitude = np.linalg.norm(new_vel)
    if new_vel_magnitude > 0:
        new_vel = new_vel / new_vel_magnitude * missile_vel_magnitude

    # Update position
    new_pos = missile_pos + new_vel * dt

    # Return new LOS
    new_los = (target_pos - new_pos)
    new_los_dist = np.linalg.norm(new_los)
    if new_los_dist > 0:
        new_los = new_los / new_los_dist

    return new_pos, new_los


def get_guidance_function(algorithm_name):
    """Return the guidance function based on algorithm name."""
    algorithms = {
        "pure_pursuit": pure_pursuit_guidance,
        "lead_pursuit": lead_pursuit_guidance,
        "proportional_navigation": proportional_navigation_guidance,
        "augmented_pn": augmented_pn_guidance,
    }
    return algorithms.get(algorithm_name, pure_pursuit_guidance)


# ============================================================================
# PHYSICS FUNCTIONS
# ============================================================================

def apply_turn_rate_limit(current_direction, desired_direction, dt):
    """
    Limit the rate at which the missile can change its heading.

    Parameters:
    - current_direction: current velocity unit vector
    - desired_direction: desired velocity unit vector (from guidance)
    - dt: time step

    Returns:
    - new_direction: limited direction unit vector
    """
    if not ENABLE_PHYSICS or not ENABLE_TURN_LIMIT:
        return desired_direction

    # Calculate angle between current and desired direction
    dot_product = np.clip(np.dot(current_direction, desired_direction), -1.0, 1.0)
    angle = np.arccos(dot_product)

    # Maximum angle change in this time step
    max_angle_change = MAX_TURN_RATE_RAD * dt

    if angle <= max_angle_change:
        # Can achieve desired direction
        return desired_direction

    # Need to limit the turn
    # Calculate rotation axis (perpendicular to both vectors)
    rotation_axis = np.cross(current_direction, desired_direction)
    axis_norm = np.linalg.norm(rotation_axis)

    if axis_norm < 1e-10:
        # Vectors are parallel or anti-parallel
        return desired_direction

    rotation_axis = rotation_axis / axis_norm

    # Rotate current direction by max_angle_change toward desired direction
    # Using Rodrigues' rotation formula
    cos_angle = np.cos(max_angle_change)
    sin_angle = np.sin(max_angle_change)

    new_direction = (current_direction * cos_angle +
                     np.cross(rotation_axis, current_direction) * sin_angle +
                     rotation_axis * np.dot(rotation_axis, current_direction) * (1 - cos_angle))

    # Normalize
    new_direction = new_direction / np.linalg.norm(new_direction)

    return new_direction


def apply_gravity(velocity, dt):
    """
    Apply gravitational acceleration to velocity.

    Parameters:
    - velocity: current velocity vector [vx, vy, vz]
    - dt: time step

    Returns:
    - new_velocity: velocity after gravity applied
    """
    if not ENABLE_PHYSICS or not ENABLE_GRAVITY:
        return velocity

    # Gravity acts in -Z direction
    gravity_accel = np.array([0, 0, -GRAVITY])
    new_velocity = velocity + gravity_accel * dt

    return new_velocity


def apply_drag(velocity, dt):
    """
    Apply aerodynamic drag to velocity.
    Drag force is proportional to v^2 and opposite to velocity direction.

    Parameters:
    - velocity: current velocity vector
    - dt: time step

    Returns:
    - new_velocity: velocity after drag applied
    """
    if not ENABLE_PHYSICS or not ENABLE_DRAG:
        return velocity

    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return velocity

    # Drag acceleration magnitude (opposite to velocity)
    drag_accel_magnitude = DRAG_COEFFICIENT * speed * speed

    # Drag acceleration vector (opposite to velocity direction)
    velocity_dir = velocity / speed
    drag_accel = -drag_accel_magnitude * velocity_dir

    new_velocity = velocity + drag_accel * dt

    return new_velocity


def apply_thrust(velocity, direction, dt, time_since_launch):
    """
    Apply thrust acceleration if fuel is available.

    Parameters:
    - velocity: current velocity vector
    - direction: thrust direction (unit vector)
    - dt: time step
    - time_since_launch: time since missile launch

    Returns:
    - new_velocity: velocity after thrust applied
    """
    if not ENABLE_PHYSICS or not ENABLE_FUEL:
        return velocity

    if time_since_launch > FUEL_DURATION:
        # No more fuel
        return velocity

    # Apply thrust in the direction of travel
    thrust_accel = direction * THRUST_ACCELERATION
    new_velocity = velocity + thrust_accel * dt

    return new_velocity


def apply_physics(position, velocity, desired_direction, dt, time_since_launch):
    """
    Apply all physics effects to the missile.

    Parameters:
    - position: current position
    - velocity: current velocity vector
    - desired_direction: desired direction from guidance (unit vector)
    - dt: time step
    - time_since_launch: time since missile launch

    Returns:
    - new_position: updated position
    - new_velocity: updated velocity
    - is_alive: whether missile is still functional
    """
    if not ENABLE_PHYSICS:
        # No physics - just move in desired direction at constant speed
        speed = np.linalg.norm(velocity)
        new_velocity = desired_direction * speed
        new_position = position + new_velocity * dt
        return new_position, new_velocity, True

    # Current state
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        current_direction = desired_direction
    else:
        current_direction = velocity / speed

    # 1. Apply turn rate limit to get actual direction
    actual_direction = apply_turn_rate_limit(current_direction, desired_direction, dt)

    # 2. Start with current velocity magnitude in new direction
    new_velocity = actual_direction * speed

    # 3. Apply thrust (if fuel available)
    new_velocity = apply_thrust(new_velocity, actual_direction, dt, time_since_launch)

    # 4. Apply gravity
    new_velocity = apply_gravity(new_velocity, dt)

    # 5. Apply drag
    new_velocity = apply_drag(new_velocity, dt)

    # 6. Check if missile is still alive (minimum speed)
    new_speed = np.linalg.norm(new_velocity)
    is_alive = new_speed >= MIN_MISSILE_SPEED

    # 7. Update position
    new_position = position + new_velocity * dt

    return new_position, new_velocity, is_alive


# ============================================================================
# EVASIVE MANEUVER FUNCTIONS
# ============================================================================

# Random seed for reproducible evasion patterns
np.random.seed(42)
jink_directions = np.random.choice([-1, 1], size=1000)  # Pre-generate jink directions
random_pattern_switches = np.random.choice(['jinking', 'weave', 'barrel_roll'], size=100)

def calculate_evasion_offset(t, base_position, base_velocity, missile_position, evasion_start_time):
    """
    Calculate evasion offset to add to the base trajectory.

    Parameters:
    - t: current time
    - base_position: position from base trajectory
    - base_velocity: velocity direction from base trajectory
    - missile_position: current missile position
    - evasion_start_time: time when evasion started

    Returns:
    - offset: 3D offset vector to add to base position
    """
    if not ENABLE_EVASION or EVASION_PATTERN == "none":
        return np.zeros(3)

    # Check if missile is close enough to trigger evasion
    if missile_position is not None:
        distance_to_missile = np.linalg.norm(base_position - missile_position)
        if distance_to_missile > EVASION_START_DISTANCE:
            return np.zeros(3)

    # Time since evasion started
    if evasion_start_time is None:
        return np.zeros(3)
    evasion_time = t - evasion_start_time

    # Calculate perpendicular directions to velocity
    vel_norm = np.linalg.norm(base_velocity)
    if vel_norm < 1e-6:
        return np.zeros(3)

    vel_unit = base_velocity / vel_norm

    # Find perpendicular vectors (lateral and vertical relative to velocity)
    if abs(vel_unit[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])

    lateral = np.cross(vel_unit, up)
    lateral = lateral / np.linalg.norm(lateral)
    vertical = np.cross(lateral, vel_unit)
    vertical = vertical / np.linalg.norm(vertical)

    offset = np.zeros(3)
    pattern = EVASION_PATTERN

    # Handle random pattern switching
    if pattern == "random":
        pattern_idx = int(evasion_time / 3.0) % len(random_pattern_switches)
        pattern = random_pattern_switches[pattern_idx]

    if pattern == "jinking":
        # Jinking: random lateral movements
        jink_idx = int(evasion_time / JINK_INTERVAL) % len(jink_directions)
        direction = jink_directions[jink_idx]

        # Smooth transition between jinks using sine
        phase = (evasion_time % JINK_INTERVAL) / JINK_INTERVAL
        smooth_factor = np.sin(phase * np.pi)

        amplitude = JINK_AMPLITUDE * EVASION_INTENSITY
        offset = lateral * direction * amplitude * smooth_factor

        # Add some vertical component too
        offset += vertical * direction * 0.3 * amplitude * smooth_factor

    elif pattern == "barrel_roll":
        # Barrel roll: spiral motion around velocity axis
        roll_angle = 2 * np.pi * BARREL_ROLL_RATE * evasion_time
        radius = BARREL_ROLL_RADIUS * EVASION_INTENSITY

        offset = (lateral * np.cos(roll_angle) + vertical * np.sin(roll_angle)) * radius

    elif pattern == "break_turn":
        # Break turn: hard turn when missile is very close
        if missile_position is not None:
            distance = np.linalg.norm(base_position - missile_position)
            if distance < BREAK_TURN_DISTANCE:
                # Determine break direction (away from missile)
                to_missile = missile_position - base_position
                to_missile_norm = to_missile / np.linalg.norm(to_missile)

                # Break perpendicular to missile direction
                break_dir = np.cross(vel_unit, to_missile_norm)
                if np.linalg.norm(break_dir) > 1e-6:
                    break_dir = break_dir / np.linalg.norm(break_dir)
                else:
                    break_dir = lateral

                # Intensity increases as missile gets closer
                intensity = (BREAK_TURN_DISTANCE - distance) / BREAK_TURN_DISTANCE
                turn_amount = np.radians(BREAK_TURN_RATE) * evasion_time * EVASION_INTENSITY
                offset = break_dir * np.sin(turn_amount) * 1000 * intensity

    elif pattern == "weave":
        # Weave: sinusoidal lateral movement
        amplitude = WEAVE_AMPLITUDE * EVASION_INTENSITY
        phase = 2 * np.pi * WEAVE_FREQUENCY * evasion_time

        offset = lateral * np.sin(phase) * amplitude
        # Add vertical weave component (out of phase)
        offset += vertical * np.sin(phase + np.pi/3) * amplitude * 0.5

    elif pattern == "split_s":
        # Split-S: dive and pull through
        # Phase 1: roll inverted (first 2 seconds)
        # Phase 2: pull through dive (2-5 seconds)
        # Phase 3: level out in opposite direction

        if evasion_time < 2.0:
            # Rolling phase
            roll_progress = evasion_time / 2.0
            offset = vertical * (-500 * roll_progress * EVASION_INTENSITY)
        elif evasion_time < 5.0:
            # Diving phase
            dive_time = evasion_time - 2.0
            dive_progress = dive_time / 3.0
            offset = vertical * (-500 - 1000 * dive_progress) * EVASION_INTENSITY
            offset += lateral * 200 * np.sin(dive_progress * np.pi) * EVASION_INTENSITY
        else:
            # Level out (maintain offset)
            offset = vertical * (-1500) * EVASION_INTENSITY

    return offset


# ============================================================================
# COUNTERMEASURES SYSTEM
# ============================================================================

class CountermeasuresSystem:
    """Manages countermeasures (flares, chaff, ECM) for the target aircraft."""

    def __init__(self):
        self.flares_remaining = FLARE_COUNT
        self.chaff_remaining = CHAFF_COUNT
        self.last_flare_time = -FLARE_DEPLOY_INTERVAL
        self.last_chaff_time = -CHAFF_DEPLOY_INTERVAL

        # Active countermeasures
        self.active_flares = []  # List of (position, deploy_time, velocity)
        self.active_chaff = []   # List of (position, deploy_time)

        # Missile states affected by countermeasures
        self.decoyed_missiles = set()  # Missiles that are chasing decoys
        self.decoyed_times = {}  # Missile index -> simulation time when decoyed
        self.decoyed_indices = {}  # Missile index -> frame index when decoyed
        self.jammed_missiles = set()   # Missiles affected by ECM

        # Random generator for countermeasure effects
        self.rng = np.random.default_rng(42)

    def update(self, t, aircraft_pos, aircraft_vel, missile_positions, missile_indices, guidance_types, dt, frame_idx=0):
        """
        Update countermeasures state and check for deployment.

        Returns:
        - decoyed_missiles: set of missile indices that are chasing decoys
        - ecm_noise: dict of missile_index -> noise vector to add to guidance
        """
        # Update active flares (they fall and eventually expire)
        self._update_active_flares(t, dt)

        # Update active chaff (expires over time)
        self._update_active_chaff(t)

        # Check if we should deploy countermeasures
        self._check_and_deploy(t, aircraft_pos, aircraft_vel, missile_positions, missile_indices, guidance_types, frame_idx)

        # Calculate ECM effects
        ecm_noise = {}
        if ENABLE_ECM and ENABLE_COUNTERMEASURES:
            for idx, m_pos in zip(missile_indices, missile_positions):
                if idx in self.decoyed_missiles:
                    continue  # Already decoyed, no ECM needed

                distance = np.linalg.norm(aircraft_pos - m_pos)
                if distance < ECM_RANGE:
                    # Apply ECM jamming
                    guidance_type = guidance_types.get(idx, "proportional_navigation")
                    susceptibility = CM_SUSCEPTIBILITY.get(guidance_type, {}).get("ecm", 0.5)

                    if self.rng.random() < ECM_EFFECTIVENESS * susceptibility:
                        # Generate random noise for guidance
                        noise_magnitude = ECM_GUIDANCE_NOISE * susceptibility
                        noise = self.rng.normal(0, noise_magnitude, 3)
                        ecm_noise[idx] = noise

        return self.decoyed_missiles, ecm_noise

    def _update_active_flares(self, t, dt):
        """Update flare positions (they fall) and remove expired ones."""
        updated_flares = []
        for pos, deploy_time, vel in self.active_flares:
            if t - deploy_time < FLARE_DURATION:
                # Flare is still active, update position (falling)
                new_pos = pos + vel * dt
                new_vel = vel.copy()
                new_vel[2] -= FLARE_FALL_RATE * dt  # Fall faster over time
                updated_flares.append((new_pos, deploy_time, new_vel))
        self.active_flares = updated_flares

    def _update_active_chaff(self, t):
        """Remove expired chaff clouds."""
        self.active_chaff = [(pos, deploy_time) for pos, deploy_time in self.active_chaff
                             if t - deploy_time < CHAFF_DURATION]

    def _check_and_deploy(self, t, aircraft_pos, aircraft_vel, missile_positions, missile_indices, guidance_types, frame_idx=0):
        """Check if countermeasures should be deployed and deploy them."""
        for idx, m_pos in zip(missile_indices, missile_positions):
            if idx in self.decoyed_missiles:
                continue  # Already dealt with this missile

            distance = np.linalg.norm(aircraft_pos - m_pos)
            guidance_type = guidance_types.get(idx, "proportional_navigation")

            # Check flare deployment
            if (ENABLE_FLARES and ENABLE_COUNTERMEASURES and
                self.flares_remaining > 0 and
                distance < FLARE_DEPLOY_DISTANCE and
                t - self.last_flare_time >= FLARE_DEPLOY_INTERVAL):

                susceptibility = CM_SUSCEPTIBILITY.get(guidance_type, {}).get("flare", 0.5)
                if self.rng.random() < FLARE_EFFECTIVENESS * susceptibility:
                    # Flare successfully decoys the missile - it explodes on flare
                    self.decoyed_missiles.add(idx)
                    self.decoyed_times[idx] = t  # Record time when decoyed
                    self.decoyed_indices[idx] = frame_idx  # Record frame index when decoyed
                    print(f"t={t:.2f}s: Flare deployed - Missile {idx+1} EXPLODED on FLARE!")

                # Deploy flare regardless of success
                flare_vel = aircraft_vel.copy() * 0.3  # Flare starts with some aircraft velocity
                flare_vel[2] -= FLARE_FALL_RATE  # Initial downward velocity
                self.active_flares.append((aircraft_pos.copy(), t, flare_vel))
                self.flares_remaining -= 1
                self.last_flare_time = t

            # Check chaff deployment
            if (ENABLE_CHAFF and ENABLE_COUNTERMEASURES and
                self.chaff_remaining > 0 and
                distance < CHAFF_DEPLOY_DISTANCE and
                t - self.last_chaff_time >= CHAFF_DEPLOY_INTERVAL):

                susceptibility = CM_SUSCEPTIBILITY.get(guidance_type, {}).get("chaff", 0.5)
                if self.rng.random() < CHAFF_EFFECTIVENESS * susceptibility:
                    # Chaff successfully confuses the missile
                    self.decoyed_missiles.add(idx)
                    print(f"t={t:.2f}s: Chaff deployed - Missile {idx+1} lost lock!")

                self.active_chaff.append((aircraft_pos.copy(), t))
                self.chaff_remaining -= 1
                self.last_chaff_time = t

    def get_decoy_position(self, missile_idx, t):
        """Get the position a decoyed missile should track (flare or chaff)."""
        # Find most recent flare
        if self.active_flares:
            return self.active_flares[-1][0]  # Return most recent flare position
        elif self.active_chaff:
            return self.active_chaff[-1][0]  # Return most recent chaff position
        return None

    def get_status(self):
        """Get countermeasures status for display."""
        return {
            "flares": self.flares_remaining,
            "chaff": self.chaff_remaining,
            "active_flares": len(self.active_flares),
            "active_chaff": len(self.active_chaff),
            "decoyed": len(self.decoyed_missiles)
        }


# ============================================================================
# MONTE CARLO SIMULATION FUNCTIONS
# ============================================================================

def run_single_simulation(params, verbose=False):
    """
    Run a single simulation with given parameters.

    Parameters:
    - params: dict with simulation parameters (missile_pos, target_pos, guidance, evasion, etc.)
    - verbose: print progress info

    Returns:
    - results: dict with simulation outcomes (hit, miss_distance, time_to_intercept, etc.)
    """
    global curve_start_x, curve_start_y, curve_start_z, curve_initialized
    global straight2_start_x, straight2_start_y, straight2_start_z, straight2_initialized
    global center_x, center_y, center_z

    # Reset initialization flags
    curve_initialized = False
    straight2_initialized = False

    # Extract parameters
    sim_missile_pos = params.get("missile_pos", missile_start_loc.copy())
    sim_target_pos = params.get("target_pos", aircraft_start_loc.copy())
    sim_guidance = params.get("guidance", GUIDANCE_ALGORITHM)
    sim_evasion = params.get("evasion", EVASION_PATTERN)
    sim_miss_vel = params.get("missile_vel", miss_vel)
    sim_targ_vel = params.get("target_vel", targ_vel)

    # Time array
    times = np.arange(0, tmax, dt)
    n_points = len(times)

    # Generate base target trajectory
    base_target_states = np.zeros((n_points, 3))

    # Temporarily modify aircraft_start_loc for this simulation
    original_aircraft_start = aircraft_start_loc.copy()

    for i in range(n_points):
        t = times[i]
        # Calculate position with offset
        if 0 <= t <= Straight_time:
            x = sim_target_pos[0] + sim_targ_vel * t
            y = sim_target_pos[1]
            z = sim_target_pos[2]
        else:
            # For curve and straight2, use target_location function with initialized values
            base_target_states[i] = target_location(t, base_target_states[:i])
            continue
        base_target_states[i] = np.array([x, y, z])

    # Calculate velocities
    base_target_velocities = np.zeros((n_points, 3))
    for i in range(1, n_points):
        base_target_velocities[i] = (base_target_states[i] - base_target_states[i-1]) / dt
    base_target_velocities[0] = base_target_velocities[1]

    # Initialize arrays
    target_states = np.zeros((n_points, 3))
    target_velocities = np.zeros((n_points, 3))
    target_accelerations = np.zeros((n_points, 3))

    missile_states = np.zeros((n_points, 3))
    missile_velocities = np.zeros((n_points, 3))
    missile_speeds = np.zeros(n_points)

    # Initial conditions
    target_states[0] = base_target_states[0]
    target_velocities[0] = base_target_velocities[0]

    missile_states[0] = sim_missile_pos
    initial_direction = base_target_states[0] - sim_missile_pos
    if np.linalg.norm(initial_direction) > 0:
        initial_direction = initial_direction / np.linalg.norm(initial_direction)
    else:
        initial_direction = np.array([1, 0, 0])
    missile_velocities[0] = initial_direction * sim_miss_vel
    missile_speeds[0] = sim_miss_vel

    # Get guidance function
    guidance_func = get_guidance_function(sim_guidance)

    # State tracking
    prev_los = None
    missile_launched = False
    launch_index = 0
    intercept_time = None
    intercept_index = None
    intercepted = False
    missile_dead = False
    missile_dead_time = None

    # Evasion state
    evasion_started = False
    evasion_start_time = None
    original_evasion_pattern = EVASION_PATTERN

    # Initialize countermeasures
    cm_system = CountermeasuresSystem() if ENABLE_COUNTERMEASURES else None
    decoyed = False

    # Run simulation
    for i in range(1, n_points):
        t = times[i]

        # Target update
        base_pos = base_target_states[i]
        base_vel = base_target_velocities[i]

        # Check evasion
        if sim_evasion != "none" and not intercepted:
            distance = np.linalg.norm(base_pos - missile_states[i-1])
            if distance <= EVASION_START_DISTANCE and not evasion_started:
                evasion_started = True
                evasion_start_time = t

            if evasion_started:
                # Calculate evasion offset based on pattern
                evasion_offset = calculate_evasion_offset(
                    t, base_pos, base_vel, missile_states[i-1], evasion_start_time
                )
            else:
                evasion_offset = np.zeros(3)
        else:
            evasion_offset = np.zeros(3)

        target_states[i] = base_pos + evasion_offset
        target_velocities[i] = (target_states[i] - target_states[i-1]) / dt

        if i > 1:
            target_accelerations[i] = (target_velocities[i] - target_velocities[i-1]) / dt

        # Countermeasures update
        if cm_system is not None and not intercepted and not missile_dead:
            _, ecm_noise = cm_system.update(
                t, target_states[i], target_velocities[i],
                [missile_states[i-1]], [0], {0: sim_guidance}, dt
            )
            if 0 in cm_system.decoyed_missiles:
                decoyed = True

        # Missile launch
        if t >= missile_launch_time and not missile_launched:
            missile_launched = True
            launch_index = i

        if missile_launched:
            if intercepted or missile_dead:
                missile_states[i] = missile_states[i-1]
                missile_velocities[i] = missile_velocities[i-1]
                missile_speeds[i] = missile_speeds[i-1]
                continue

            # Get target position (possibly decoyed)
            if decoyed and cm_system is not None:
                decoy_pos = cm_system.get_decoy_position(0, t)
                if decoy_pos is not None:
                    target_pos = decoy_pos
                    target_vel = np.zeros(3)
                    target_accel = np.zeros(3)
                else:
                    target_pos = target_states[i]
                    target_vel = target_velocities[i]
                    target_accel = target_accelerations[i]
            else:
                target_pos = target_states[i]
                target_vel = target_velocities[i]
                target_accel = target_accelerations[i]

            # Check intercept (real target only)
            real_distance = np.linalg.norm(target_states[i] - missile_states[i-1])
            if real_distance < kill_dist and intercept_time is None and not decoyed:
                intercept_time = t
                intercept_index = i
                intercepted = True
                missile_states[i] = missile_states[i-1]
                missile_velocities[i] = missile_velocities[i-1]
                missile_speeds[i] = missile_speeds[i-1]
                continue

            # Apply guidance
            if sim_guidance == "augmented_pn":
                guided_pos, prev_los = augmented_pn_guidance(
                    missile_states[i-1], target_pos, target_vel,
                    sim_miss_vel, dt, prev_los, target_accel
                )
            else:
                guided_pos, prev_los = guidance_func(
                    missile_states[i-1], target_pos, target_vel,
                    sim_miss_vel, dt, prev_los
                )

            # Calculate desired direction
            desired_direction = guided_pos - missile_states[i-1]
            dir_norm = np.linalg.norm(desired_direction)
            if dir_norm > 1e-6:
                desired_direction = desired_direction / dir_norm
            else:
                desired_direction = prev_los if prev_los is not None else np.array([1, 0, 0])

            # Apply physics
            time_since_launch = t - times[launch_index]
            new_pos, new_vel, is_alive = apply_physics(
                missile_states[i-1],
                missile_velocities[i-1],
                desired_direction,
                dt,
                time_since_launch
            )

            if not is_alive and not missile_dead:
                missile_dead = True
                missile_dead_time = t

            missile_states[i] = new_pos
            missile_velocities[i] = new_vel
            missile_speeds[i] = np.linalg.norm(new_vel)
        else:
            missile_states[i] = sim_missile_pos
            missile_velocities[i] = missile_velocities[0]
            missile_speeds[i] = sim_miss_vel

    # Calculate final results
    final_distance = np.linalg.norm(target_states[-1] - missile_states[-1])
    min_distance = np.min([np.linalg.norm(target_states[i] - missile_states[i])
                          for i in range(n_points)])

    results = {
        "hit": intercepted,
        "decoyed": decoyed,
        "missile_dead": missile_dead,
        "intercept_time": intercept_time,
        "final_distance": final_distance,
        "min_distance": min_distance,
        "guidance": sim_guidance,
        "evasion": sim_evasion,
        "missile_pos": sim_missile_pos.copy(),
        "target_pos": sim_target_pos.copy(),
    }

    return results


def run_monte_carlo_analysis():
    """
    Run Monte Carlo analysis with multiple simulations.
    Returns statistics and generates visualizations.
    """
    print("\n" + "="*60)
    print("MONTE CARLO ANALYSIS")
    print("="*60)
    print(f"Running {MONTE_CARLO_RUNS} simulations...")

    rng = np.random.default_rng(42)  # Reproducible results
    results_list = []

    start_time = time.time()

    for run in range(MONTE_CARLO_RUNS):
        # Generate randomized parameters
        params = {}

        # Randomize missile position
        if MC_RANDOMIZE_MISSILE_POS:
            offset = rng.normal(0, MC_MISSILE_POS_VARIANCE, 3)
            params["missile_pos"] = missile_start_loc + offset
        else:
            params["missile_pos"] = missile_start_loc.copy()

        # Randomize target position
        if MC_RANDOMIZE_TARGET_POS:
            offset = rng.normal(0, MC_TARGET_POS_VARIANCE, 3)
            params["target_pos"] = aircraft_start_loc + offset
        else:
            params["target_pos"] = aircraft_start_loc.copy()

        # Randomize velocities
        if MC_RANDOMIZE_VELOCITIES:
            params["missile_vel"] = miss_vel + rng.normal(0, MC_VELOCITY_VARIANCE)
            params["target_vel"] = targ_vel + rng.normal(0, MC_VELOCITY_VARIANCE)
        else:
            params["missile_vel"] = miss_vel
            params["target_vel"] = targ_vel

        # Randomize guidance algorithm
        if MC_RANDOMIZE_GUIDANCE:
            params["guidance"] = rng.choice(MC_GUIDANCE_ALGORITHMS)
        else:
            params["guidance"] = GUIDANCE_ALGORITHM

        # Randomize evasion pattern
        if MC_RANDOMIZE_EVASION:
            params["evasion"] = rng.choice(MC_EVASION_PATTERNS)
        else:
            params["evasion"] = EVASION_PATTERN

        # Run simulation
        result = run_single_simulation(params, verbose=False)
        results_list.append(result)

        # Progress update
        if (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{MONTE_CARLO_RUNS} simulations...")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")

    # Analyze results
    hits = sum(1 for r in results_list if r["hit"])
    decoyed = sum(1 for r in results_list if r["decoyed"])
    dead = sum(1 for r in results_list if r["missile_dead"])
    misses = MONTE_CARLO_RUNS - hits - decoyed - dead

    hit_rate = hits / MONTE_CARLO_RUNS * 100
    decoy_rate = decoyed / MONTE_CARLO_RUNS * 100
    dead_rate = dead / MONTE_CARLO_RUNS * 100

    min_distances = [r["min_distance"] for r in results_list]
    final_distances = [r["final_distance"] for r in results_list]
    intercept_times = [r["intercept_time"] for r in results_list if r["intercept_time"] is not None]

    print("\n--- MONTE CARLO RESULTS ---")
    print(f"Total Runs: {MONTE_CARLO_RUNS}")
    print(f"Hits: {hits} ({hit_rate:.1f}%)")
    print(f"Decoyed: {decoyed} ({decoy_rate:.1f}%)")
    print(f"Missile Dead: {dead} ({dead_rate:.1f}%)")
    print(f"Misses: {misses} ({misses/MONTE_CARLO_RUNS*100:.1f}%)")
    print(f"\nMinimum Distance Stats:")
    print(f"  Mean: {np.mean(min_distances):.1f} m")
    print(f"  Std Dev: {np.std(min_distances):.1f} m")
    print(f"  Min: {np.min(min_distances):.1f} m")
    print(f"  Max: {np.max(min_distances):.1f} m")

    if intercept_times:
        print(f"\nIntercept Time Stats (hits only):")
        print(f"  Mean: {np.mean(intercept_times):.2f} s")
        print(f"  Std Dev: {np.std(intercept_times):.2f} s")

    # Results by guidance algorithm
    print("\n--- Results by Guidance Algorithm ---")
    for guidance in MC_GUIDANCE_ALGORITHMS:
        guidance_results = [r for r in results_list if r["guidance"] == guidance]
        if guidance_results:
            g_hits = sum(1 for r in guidance_results if r["hit"])
            g_total = len(guidance_results)
            print(f"  {guidance}: {g_hits}/{g_total} hits ({g_hits/g_total*100:.1f}%)")

    # Results by evasion pattern
    print("\n--- Results by Evasion Pattern ---")
    for evasion in MC_EVASION_PATTERNS:
        evasion_results = [r for r in results_list if r["evasion"] == evasion]
        if evasion_results:
            e_hits = sum(1 for r in evasion_results if r["hit"])
            e_total = len(evasion_results)
            print(f"  {evasion}: {e_hits}/{e_total} hits ({e_hits/e_total*100:.1f}%)")

    # Generate visualizations
    if MC_SHOW_HISTOGRAMS or MC_SHOW_SCATTER:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Monte Carlo Analysis Results ({MONTE_CARLO_RUNS} runs)', fontsize=14)

        # Histogram of minimum distances
        ax1 = axes[0, 0]
        ax1.hist(min_distances, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(kill_dist, color='red', linestyle='--', label=f'Kill Distance ({kill_dist}m)')
        ax1.set_xlabel('Minimum Distance (m)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Minimum Distances')
        ax1.legend()

        # Pie chart of outcomes
        ax2 = axes[0, 1]
        labels = ['Hit', 'Decoyed', 'Missile Dead', 'Miss']
        sizes = [hits, decoyed, dead, misses]
        colors = ['green', 'orange', 'gray', 'red']
        explode = (0.1, 0, 0, 0)
        # Filter out zero values
        non_zero = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0]
        if non_zero:
            labels, sizes, colors, explode = zip(*non_zero)
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
        ax2.set_title('Engagement Outcomes')

        # Hit rate by guidance algorithm
        ax3 = axes[1, 0]
        guidance_names = []
        guidance_hit_rates = []
        for guidance in MC_GUIDANCE_ALGORITHMS:
            guidance_results = [r for r in results_list if r["guidance"] == guidance]
            if guidance_results:
                g_hits = sum(1 for r in guidance_results if r["hit"])
                g_total = len(guidance_results)
                guidance_names.append(guidance.replace('_', '\n'))
                guidance_hit_rates.append(g_hits / g_total * 100)
        bars = ax3.bar(guidance_names, guidance_hit_rates, color=['blue', 'green', 'orange', 'red'])
        ax3.set_ylabel('Hit Rate (%)')
        ax3.set_title('Hit Rate by Guidance Algorithm')
        ax3.set_ylim(0, 100)
        for bar, rate in zip(bars, guidance_hit_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        # Hit rate by evasion pattern
        ax4 = axes[1, 1]
        evasion_names = []
        evasion_hit_rates = []
        for evasion in MC_EVASION_PATTERNS:
            evasion_results = [r for r in results_list if r["evasion"] == evasion]
            if evasion_results:
                e_hits = sum(1 for r in evasion_results if r["hit"])
                e_total = len(evasion_results)
                evasion_names.append(evasion.replace('_', '\n'))
                evasion_hit_rates.append(e_hits / e_total * 100)
        bars = ax4.bar(evasion_names, evasion_hit_rates, color=['gray', 'blue', 'green', 'orange', 'red'])
        ax4.set_ylabel('Hit Rate (%)')
        ax4.set_title('Hit Rate by Evasion Pattern')
        ax4.set_ylim(0, 100)
        for bar, rate in zip(bars, evasion_hit_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

    # Save results to CSV if requested
    if MC_SAVE_RESULTS:
        import csv
        with open(MC_RESULTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
            writer.writeheader()
            for result in results_list:
                # Convert numpy arrays to lists for CSV
                row = result.copy()
                row["missile_pos"] = list(row["missile_pos"])
                row["target_pos"] = list(row["target_pos"])
                writer.writerow(row)
        print(f"\nResults saved to {MC_RESULTS_FILE}")

    return results_list


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Check if Monte Carlo mode is enabled
if ENABLE_MONTE_CARLO:
    # Run Monte Carlo analysis and exit
    mc_results = run_monte_carlo_analysis()
    print("\nMonte Carlo analysis complete. Exiting...")
    import sys
    sys.exit(0)

# ============================================================================
# STANDARD SIMULATION MODE
# ============================================================================
# If we reach here, we're running standard single simulation with visualization

# ============================================================================
# GENERATE COUPLED TARGET AND MISSILE TRAJECTORIES
# ============================================================================
# Reset initialization flags before generating trajectory
curve_initialized = False
straight2_initialized = False

# Time array
times = np.arange(0, tmax, dt)
n_points = len(times)

# First pass: Generate base target trajectory (without evasion)
base_target_states = np.zeros((n_points, 3))
for i in range(n_points):
    t = times[i]
    base_target_states[i] = target_location(t, base_target_states[:i])

# Calculate base velocities
base_target_velocities = np.zeros((n_points, 3))
for i in range(1, n_points):
    base_target_velocities[i] = (base_target_states[i] - base_target_states[i-1]) / dt
base_target_velocities[0] = base_target_velocities[1]

print(f"Generated {n_points} base trajectory points over {tmax:.1f} seconds")
print(f"Aircraft start: ({aircraft_start_loc[0]:.1f}, {aircraft_start_loc[1]:.1f}, {aircraft_start_loc[2]:.1f})")
print(f"Curve start: ({curve_start_x:.1f}, {curve_start_y:.1f}, {curve_start_z:.1f})")
print(f"Straight2 start: ({straight2_start_x:.1f}, {straight2_start_y:.1f}, {straight2_start_z:.1f})")
print(f"\nUsing guidance algorithm: {GUIDANCE_ALGORITHM}")
if GUIDANCE_ALGORITHM in ["proportional_navigation", "augmented_pn"]:
    print(f"Navigation constant (N): {PN_CONSTANT}")

# Print physics settings
if ENABLE_PHYSICS:
    print(f"\nPhysics enabled:")
    print(f"  - Gravity: {'ON' if ENABLE_GRAVITY else 'OFF'} ({GRAVITY} m/s²)")
    print(f"  - Drag: {'ON' if ENABLE_DRAG else 'OFF'} (coeff: {DRAG_COEFFICIENT})")
    print(f"  - Turn limit: {'ON' if ENABLE_TURN_LIMIT else 'OFF'} ({MAX_TURN_RATE}°/s)")
    print(f"  - Fuel: {'ON' if ENABLE_FUEL else 'OFF'} ({FUEL_DURATION}s, {THRUST_ACCELERATION} m/s²)")
else:
    print(f"\nPhysics disabled (ideal motion)")

# Print evasion settings
if ENABLE_EVASION and EVASION_PATTERN != "none":
    print(f"\nEvasion enabled:")
    print(f"  - Pattern: {EVASION_PATTERN}")
    print(f"  - Start distance: {EVASION_START_DISTANCE} m")
    print(f"  - Intensity: {EVASION_INTENSITY}")
else:
    print(f"\nEvasion disabled")

# ============================================================================
# COUPLED SIMULATION (Target with evasion + Multiple Missiles)
# ============================================================================

# Determine number of missiles
if ENABLE_MULTI_MISSILE:
    num_missiles = min(NUM_MISSILES, len(MISSILE_CONFIGS))
    print(f"\nMulti-missile mode: {num_missiles} missiles")
else:
    num_missiles = 1

# Initialize target arrays
target_states = np.zeros((n_points, 3))
target_velocities = np.zeros((n_points, 3))
target_accelerations = np.zeros((n_points, 3))

# Initialize missile arrays (now supporting multiple missiles)
all_missile_states = np.zeros((num_missiles, n_points, 3))
all_missile_velocities = np.zeros((num_missiles, n_points, 3))
all_missile_speeds = np.zeros((num_missiles, n_points))

# Missile state tracking for each missile
missile_launched = [False] * num_missiles
missile_launch_indices = [0] * num_missiles
intercept_times = [None] * num_missiles
intercept_indices = [None] * num_missiles
intercepted = [False] * num_missiles
missile_dead = [False] * num_missiles
missile_dead_times = [None] * num_missiles
prev_los_list = [None] * num_missiles
guidance_funcs = []

# Initialize each missile
for m in range(num_missiles):
    if ENABLE_MULTI_MISSILE:
        config = MISSILE_CONFIGS[m]
        start_pos = config["start_pos"]
        guidance_name = config["guidance"]
    else:
        start_pos = missile_start_loc
        guidance_name = GUIDANCE_ALGORITHM

    all_missile_states[m, 0] = start_pos
    guidance_funcs.append(get_guidance_function(guidance_name))

    # Initial velocity pointing toward target
    initial_direction = base_target_states[0] - start_pos
    if np.linalg.norm(initial_direction) > 0:
        initial_direction = initial_direction / np.linalg.norm(initial_direction)
    else:
        initial_direction = np.array([1, 0, 0])
    all_missile_velocities[m, 0] = initial_direction * miss_vel
    all_missile_speeds[m, 0] = miss_vel

    if ENABLE_MULTI_MISSILE:
        print(f"  Missile {m+1}: {guidance_name} from {start_pos}")

# Initial target conditions
target_states[0] = base_target_states[0]
target_velocities[0] = base_target_velocities[0]

# Evasion state
evasion_started = False
evasion_start_time = None
any_intercepted = False

# Initialize countermeasures system
cm_system = CountermeasuresSystem() if ENABLE_COUNTERMEASURES else None
decoyed_missiles = set()
ecm_noise = {}

# Track flare count at each frame for animation display
flare_counts_per_frame = np.full(n_points, FLARE_COUNT, dtype=int)

# Build guidance types dictionary for countermeasures
guidance_types = {}
for m in range(num_missiles):
    if ENABLE_MULTI_MISSILE:
        guidance_types[m] = MISSILE_CONFIGS[m]["guidance"]
    else:
        guidance_types[m] = GUIDANCE_ALGORITHM

# Print countermeasures status
if ENABLE_COUNTERMEASURES:
    print(f"\nCountermeasures enabled:")
    print(f"  - Flares: {FLARE_COUNT} (deploy at {FLARE_DEPLOY_DISTANCE}m)")
    print(f"  - Chaff: {CHAFF_COUNT} (deploy at {CHAFF_DEPLOY_DISTANCE}m)")
    print(f"  - ECM: {'ON' if ENABLE_ECM else 'OFF'} (range: {ECM_RANGE}m)")

# For backwards compatibility - create single missile reference
missile_states = all_missile_states[0]
missile_velocities = all_missile_velocities[0]
missile_speeds = all_missile_speeds[0]

for i in range(1, n_points):
    t = times[i]

    # ========== TARGET UPDATE ==========
    base_pos = base_target_states[i]
    base_vel = base_target_velocities[i]

    # Check if evasion should start (based on closest missile)
    if ENABLE_EVASION and EVASION_PATTERN != "none" and not any_intercepted:
        min_distance = float('inf')
        closest_missile_pos = None
        for m in range(num_missiles):
            if missile_launched[m] and not intercepted[m]:
                dist = np.linalg.norm(base_pos - all_missile_states[m, i-1])
                if dist < min_distance:
                    min_distance = dist
                    closest_missile_pos = all_missile_states[m, i-1]

        if closest_missile_pos is not None and min_distance <= EVASION_START_DISTANCE and not evasion_started:
            evasion_started = True
            evasion_start_time = t
            print(f"Evasion started at t = {t:.2f}s (closest missile at {min_distance:.0f}m)")

    # Calculate evasion offset
    if evasion_started and closest_missile_pos is not None:
        evasion_offset = calculate_evasion_offset(
            t, base_pos, base_vel, closest_missile_pos, evasion_start_time
        )
    else:
        evasion_offset = np.zeros(3)

    # Apply evasion to target position
    target_states[i] = base_pos + evasion_offset
    target_velocities[i] = (target_states[i] - target_states[i-1]) / dt

    if i > 1:
        target_accelerations[i] = (target_velocities[i] - target_velocities[i-1]) / dt
    else:
        target_accelerations[i] = np.zeros(3)

    # ========== COUNTERMEASURES UPDATE ==========
    if cm_system is not None:
        # Gather active missile positions
        active_missile_positions = []
        active_missile_indices = []
        for m in range(num_missiles):
            if missile_launched[m] and not intercepted[m] and not missile_dead[m]:
                active_missile_positions.append(all_missile_states[m, i-1])
                active_missile_indices.append(m)

        if active_missile_positions:
            decoyed_missiles, ecm_noise = cm_system.update(
                t, target_states[i], target_velocities[i],
                active_missile_positions, active_missile_indices,
                guidance_types, dt, frame_idx=i
            )
        # Track flare count at this frame
        flare_counts_per_frame[i] = cm_system.flares_remaining

    # ========== MISSILES UPDATE ==========
    for m in range(num_missiles):
        # Sequential launch logic - only launch next missile if previous failed
        if SEQUENTIAL_LAUNCH and m > 0:
            prev_m = m - 1
            # Check if previous missile failed (decoyed or dead, but not intercepted)
            prev_failed = (missile_launched[prev_m] and
                          (prev_m in decoyed_missiles or missile_dead[prev_m]) and
                          not intercepted[prev_m])
            if not prev_failed:
                # Previous missile still active or hasn't launched yet
                all_missile_states[m, i] = all_missile_states[m, i-1] if i > 0 else MISSILE_CONFIGS[m]["start_pos"]
                all_missile_velocities[m, i] = all_missile_velocities[m, 0]
                all_missile_speeds[m, i] = miss_vel
                continue

        # Get launch time for this missile
        if ENABLE_MULTI_MISSILE:
            launch_time = MISSILE_CONFIGS[m]["launch_time"]
            if SALVO_MODE:
                launch_time = missile_launch_time + m * SALVO_INTERVAL
        else:
            launch_time = missile_launch_time

        # Check if missile should launch
        if t >= launch_time and not missile_launched[m]:
            missile_launched[m] = True
            missile_launch_indices[m] = i
            missile_name = MISSILE_CONFIGS[m].get("name", f"Missile {m+1}") if ENABLE_MULTI_MISSILE else "Missile"
            if ENABLE_MULTI_MISSILE:
                print(f"[LAUNCH] {missile_name} LAUNCHED from Launcher {m+1}!")
            else:
                print(f"[LAUNCH] Missile launched!")

        if missile_launched[m]:
            # Missile holds position after intercept, death, or decoyed (exploded on flare)
            if intercepted[m] or missile_dead[m] or m in decoyed_missiles:
                all_missile_states[m, i] = all_missile_states[m, i-1]
                all_missile_velocities[m, i] = all_missile_velocities[m, i-1]
                all_missile_speeds[m, i] = all_missile_speeds[m, i-1]
                continue

            # Get target info
            target_pos = target_states[i]
            target_vel = target_velocities[i]
            target_accel = target_accelerations[i]

            # Apply ECM noise to target position if affected
            if m in ecm_noise:
                noise = ecm_noise[m]
                target_pos = target_pos + noise * 500  # Scale noise to position offset

            # Check for intercept (only against real target, not decoys)
            real_distance = np.linalg.norm(target_states[i] - all_missile_states[m, i-1])
            if real_distance < kill_dist and intercept_times[m] is None and m not in decoyed_missiles:
                intercept_times[m] = t
                intercept_indices[m] = i
                intercepted[m] = True
                any_intercepted = True
                if ENABLE_MULTI_MISSILE:
                    print(f"Missile {m+1} intercept at t = {t:.2f}s, distance = {real_distance:.1f}m")
                else:
                    print(f"Intercept at t = {t:.2f}s, distance = {real_distance:.1f}m")
                all_missile_states[m, i] = all_missile_states[m, i-1]
                all_missile_velocities[m, i] = all_missile_velocities[m, i-1]
                all_missile_speeds[m, i] = all_missile_speeds[m, i-1]
                continue

            # Apply guidance algorithm
            guidance_name = MISSILE_CONFIGS[m]["guidance"] if ENABLE_MULTI_MISSILE else GUIDANCE_ALGORITHM
            if guidance_name == "augmented_pn":
                guided_pos, prev_los_list[m] = augmented_pn_guidance(
                    all_missile_states[m, i-1], target_pos, target_vel,
                    miss_vel, dt, prev_los_list[m], target_accel
                )
            else:
                guided_pos, prev_los_list[m] = guidance_funcs[m](
                    all_missile_states[m, i-1], target_pos, target_vel,
                    miss_vel, dt, prev_los_list[m]
                )

            # Calculate desired direction from guidance
            desired_direction = guided_pos - all_missile_states[m, i-1]
            dir_norm = np.linalg.norm(desired_direction)
            if dir_norm > 1e-6:
                desired_direction = desired_direction / dir_norm
            else:
                desired_direction = prev_los_list[m] if prev_los_list[m] is not None else np.array([1, 0, 0])

            # Apply physics
            time_since_launch = t - times[missile_launch_indices[m]]
            new_pos, new_vel, is_alive = apply_physics(
                all_missile_states[m, i-1],
                all_missile_velocities[m, i-1],
                desired_direction,
                dt,
                time_since_launch
            )

            # Check if missile died (too slow)
            if not is_alive and not missile_dead[m]:
                missile_dead[m] = True
                missile_dead_times[m] = t
                if ENABLE_MULTI_MISSILE:
                    print(f"Missile {m+1} lost energy at t = {t:.2f}s (speed below {MIN_MISSILE_SPEED} m/s)")
                else:
                    print(f"Missile lost energy at t = {t:.2f}s (speed below {MIN_MISSILE_SPEED} m/s)")

            all_missile_states[m, i] = new_pos
            all_missile_velocities[m, i] = new_vel
            all_missile_speeds[m, i] = np.linalg.norm(new_vel)
        else:
            # Missile hasn't launched yet, stays at starting position
            if ENABLE_MULTI_MISSILE:
                all_missile_states[m, i] = MISSILE_CONFIGS[m]["start_pos"]
            else:
                all_missile_states[m, i] = missile_start_loc
            all_missile_velocities[m, i] = all_missile_velocities[m, 0]
            all_missile_speeds[m, i] = miss_vel

# Update references for backwards compatibility
missile_states = all_missile_states[0]
missile_velocities = all_missile_velocities[0]
missile_speeds = all_missile_speeds[0]
intercept_time = intercept_times[0]
intercept_index = intercept_indices[0]

# Calculate final miss distances
print(f"\n--- Simulation Results ---")
for m in range(num_missiles):
    final_dist = np.linalg.norm(target_states[-1] - all_missile_states[m, -1])
    if intercepted[m]:
        status = "HIT"
    elif m in decoyed_missiles:
        status = "DECOYED"
    elif missile_dead[m]:
        status = "DEAD"
    else:
        status = "MISS"
    if ENABLE_MULTI_MISSILE:
        print(f"Missile {m+1}: {status}, final distance: {final_dist:.1f}m")
    else:
        print(f"Result: {status}, final distance: {final_dist:.1f}m")

# Print countermeasures summary
if ENABLE_COUNTERMEASURES and cm_system is not None:
    cm_status = cm_system.get_status()
    print(f"\n--- Countermeasures Summary ---")
    print(f"Flares used: {FLARE_COUNT - cm_status['flares']} / {FLARE_COUNT}")
    print(f"Chaff used: {CHAFF_COUNT - cm_status['chaff']} / {CHAFF_COUNT}")
    print(f"Missiles decoyed: {cm_status['decoyed']} / {num_missiles}")

# ============================================================================
# CREATE 3D PLOT WITH STATUS PANEL
# ============================================================================
fig = plt.figure(figsize=(16, 10))

# Create grid layout: left panel for status, right for 3D view
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.05)

# Left panel - Status display (2D)
ax_status_panel = fig.add_subplot(gs[0])
ax_status_panel.set_xlim(0, 1)
ax_status_panel.set_ylim(0, 1)
ax_status_panel.set_facecolor('#1a1a2e')
ax_status_panel.set_xticks([])
ax_status_panel.set_yticks([])
ax_status_panel.set_title('TACTICAL STATUS', fontsize=14, fontweight='bold', color='white',
                          pad=10, backgroundcolor='#16213e')

# Right panel - 3D simulation
ax = fig.add_subplot(gs[1], projection='3d')

# Set axis limits based on all trajectories
all_trajectory_points = [target_states]
for m in range(num_missiles):
    all_trajectory_points.append(all_missile_states[m])
all_points = np.vstack(all_trajectory_points)

padding = 0.1
x_range = np.ptp(all_points[:, 0])
y_range = np.ptp(all_points[:, 1])
z_range = np.ptp(all_points[:, 2])
max_range = max(x_range, y_range, z_range)

x_center = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
y_center = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
z_center = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2

plot_radius = max_range / 2 * (1 + padding)

ax.set_xlim(x_center - plot_radius, x_center + plot_radius)
ax.set_ylim(y_center - plot_radius, y_center + plot_radius)
ax.set_zlim(z_center - plot_radius, z_center + plot_radius)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'[HEAT-SEEKER INTERCEPT SIMULATION]\n{num_missiles} SAM Launchers | Aircraft has {FLARE_COUNT} Flares')
ax.grid(True)
ax.view_init(elev=25, azim=45)

# Create animation artists - Jet as CIRCLE marker (changes to STAR when destroyed)
target_point, = ax.plot([], [], [], marker='o', color='blue', markersize=12, label='JET', linestyle='None')
target_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5, label='Jet Trail')

# Create artists for each missile
missile_points = []
missile_trails = []
missile_colors = ['red', 'orange', 'magenta', 'cyan', 'yellow', 'lime']

for m in range(num_missiles):
    if ENABLE_MULTI_MISSILE:
        color = MISSILE_CONFIGS[m].get("color", missile_colors[m % len(missile_colors)])
    else:
        color = 'red'
    point, = ax.plot([], [], [], 'o', color=color, markersize=8, label=f'SAM-{m+1}' if ENABLE_MULTI_MISSILE else 'Missile')
    trail, = ax.plot([], [], [], '-', color=color, linewidth=1.5, alpha=0.5)
    missile_points.append(point)
    missile_trails.append(trail)

# ============================================================================
# STATUS PANEL UI ELEMENTS (Left Panel)
# ============================================================================
# Draw status boxes on the left panel

# Title box
ax_status_panel.add_patch(plt.Rectangle((0.05, 0.92), 0.9, 0.06, facecolor='#0f3460', edgecolor='white', linewidth=2))
panel_title = ax_status_panel.text(0.5, 0.95, 'MISSION CONTROL', ha='center', va='center',
                                    fontsize=12, fontweight='bold', color='#00ff00', family='monospace')

# JET STATUS BOX
ax_status_panel.add_patch(plt.Rectangle((0.05, 0.78), 0.9, 0.12, facecolor='#16213e', edgecolor='#00ffff', linewidth=2))
ax_status_panel.text(0.1, 0.875, 'JET FIGHTER', fontsize=10, fontweight='bold', color='#00ffff', family='monospace')
jet_status_text = ax_status_panel.text(0.1, 0.81, 'Status: ACTIVE', fontsize=9, color='#00ff00', family='monospace')
jet_flare_text = ax_status_panel.text(0.55, 0.81, 'Flares: 2/2', fontsize=9, color='orange', family='monospace')

# SAM-1 STATUS BOX
ax_status_panel.add_patch(plt.Rectangle((0.05, 0.62), 0.9, 0.14, facecolor='#16213e', edgecolor='red', linewidth=2))
ax_status_panel.text(0.1, 0.735, 'SAM-1 [EAST]', fontsize=10, fontweight='bold', color='red', family='monospace')
sam1_status_text = ax_status_panel.text(0.1, 0.675, 'Status: READY', fontsize=9, color='yellow', family='monospace')
sam1_dist_text = ax_status_panel.text(0.1, 0.635, 'Distance: ---', fontsize=8, color='white', family='monospace')

# SAM-2 STATUS BOX
ax_status_panel.add_patch(plt.Rectangle((0.05, 0.46), 0.9, 0.14, facecolor='#16213e', edgecolor='orange', linewidth=2))
ax_status_panel.text(0.1, 0.575, 'SAM-2 [NORTH]', fontsize=10, fontweight='bold', color='orange', family='monospace')
sam2_status_text = ax_status_panel.text(0.1, 0.515, 'Status: STANDBY', fontsize=9, color='gray', family='monospace')
sam2_dist_text = ax_status_panel.text(0.1, 0.475, 'Distance: ---', fontsize=8, color='white', family='monospace')

# SAM-3 STATUS BOX
ax_status_panel.add_patch(plt.Rectangle((0.05, 0.30), 0.9, 0.14, facecolor='#16213e', edgecolor='magenta', linewidth=2))
ax_status_panel.text(0.1, 0.415, 'SAM-3 [NORTHWEST]', fontsize=10, fontweight='bold', color='magenta', family='monospace')
sam3_status_text = ax_status_panel.text(0.1, 0.355, 'Status: STANDBY', fontsize=9, color='gray', family='monospace')
sam3_dist_text = ax_status_panel.text(0.1, 0.315, 'Distance: ---', fontsize=8, color='white', family='monospace')

# MISSION RESULT BOX
ax_status_panel.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.18, facecolor='#0f3460', edgecolor='#ffff00', linewidth=2))
ax_status_panel.text(0.1, 0.225, 'MISSION STATUS', fontsize=10, fontweight='bold', color='#ffff00', family='monospace')
mission_status_text = ax_status_panel.text(0.1, 0.165, 'IN PROGRESS...', fontsize=11, color='white', family='monospace')
mission_detail_text = ax_status_panel.text(0.1, 0.105, '', fontsize=9, color='#aaaaaa', family='monospace')

# Store all status text elements for updating
status_texts = {
    'jet_status': jet_status_text,
    'jet_flare': jet_flare_text,
    'sam1_status': sam1_status_text,
    'sam1_dist': sam1_dist_text,
    'sam2_status': sam2_status_text,
    'sam2_dist': sam2_dist_text,
    'sam3_status': sam3_status_text,
    'sam3_dist': sam3_dist_text,
    'mission_status': mission_status_text,
    'mission_detail': mission_detail_text,
}

# Intercept notification text (large, centered) - on 3D view
intercept_text = ax.text2D(0.5, 0.5, '', transform=ax.transAxes, fontsize=28,
                           color='red', fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

# Decoyed notification text (shows when missile hits flare)
decoyed_text = ax.text2D(0.5, 0.7, '', transform=ax.transAxes, fontsize=20,
                         color='orange', fontweight='bold', ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# Pause/countdown text
pause_text = ax.text2D(0.5, 0.35, '', transform=ax.transAxes, fontsize=16,
                       color='white', ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))

# Global pause state for when Jet is destroyed
pause_state = {
    'paused': False,
    'pause_start_frame': 0,
    'frozen_frame': None,  # Frame to freeze at when jet destroyed
    'anim_ref': None  # Reference to animation object
}

# Add "Next Simulation" button at bottom of status panel
from matplotlib.widgets import Button
ax_next_btn = plt.axes([0.02, 0.02, 0.18, 0.04])  # Position at bottom left
btn_next_sim = Button(ax_next_btn, 'NEXT SIMULATION', color='#1a5f1a', hovercolor='#2d8f2d')
btn_next_sim.label.set_color('white')
btn_next_sim.label.set_fontweight('bold')

def on_next_simulation(event):
    """Restart the simulation by re-running the script."""
    import subprocess
    import sys
    import os
    # Close current figure and restart
    plt.close('all')
    # Restart the script
    os.execv(sys.executable, [sys.executable] + sys.argv)

btn_next_sim.on_clicked(on_next_simulation)

# Starting position markers
ax.scatter(target_states[0, 0], target_states[0, 1], target_states[0, 2],
           c='green', s=100, marker='s', label='Aircraft Start')

# Mark missile start positions
for m in range(num_missiles):
    if ENABLE_MULTI_MISSILE:
        start_pos = MISSILE_CONFIGS[m]["start_pos"]
        color = MISSILE_CONFIGS[m].get("color", missile_colors[m % len(missile_colors)])
    else:
        start_pos = missile_start_loc
        color = 'orange'
    ax.scatter(start_pos[0], start_pos[1], start_pos[2],
               c=color, s=100, marker='^', alpha=0.7)

# Mark intercept points
for m in range(num_missiles):
    if intercept_indices[m] is not None:
        idx = intercept_indices[m]
        ax.scatter(target_states[idx, 0], target_states[idx, 1],
                   target_states[idx, 2],
                   c='red', s=200, marker='*')

ax.legend(loc='upper right')

# ============================================================================
# ENHANCED VISUALIZATION ELEMENTS
# ============================================================================

# Add ground plane
if SHOW_GROUND_PLANE:
    # Create a ground plane at z=0
    ground_x = np.linspace(x_center - plot_radius, x_center + plot_radius, 10)
    ground_y = np.linspace(y_center - plot_radius, y_center + plot_radius, 10)
    ground_X, ground_Y = np.meshgrid(ground_x, ground_y)
    ground_Z = np.zeros_like(ground_X)
    ax.plot_surface(ground_X, ground_Y, ground_Z, alpha=GROUND_PLANE_ALPHA,
                    color=GROUND_PLANE_COLOR, shade=False)
    # Add grid lines on ground
    for gx in ground_x[::2]:
        ax.plot([gx, gx], [ground_y[0], ground_y[-1]], [0, 0],
                color='darkgreen', alpha=0.2, linewidth=0.5)
    for gy in ground_y[::2]:
        ax.plot([ground_x[0], ground_x[-1]], [gy, gy], [0, 0],
                color='darkgreen', alpha=0.2, linewidth=0.5)

# Create distance line artists (one per missile)
distance_lines = []
if SHOW_DISTANCE_LINES:
    for m in range(num_missiles):
        line, = ax.plot([], [], [], color=DISTANCE_LINE_COLOR,
                        alpha=DISTANCE_LINE_ALPHA, linestyle='--', linewidth=1)
        distance_lines.append(line)

# Create altitude line artists
altitude_lines = []
if SHOW_ALTITUDE_LINES:
    # One for target
    target_alt_line, = ax.plot([], [], [], color='blue', alpha=ALTITUDE_LINE_ALPHA,
                               linestyle=':', linewidth=1)
    altitude_lines.append(target_alt_line)
    # One for each missile
    for m in range(num_missiles):
        if ENABLE_MULTI_MISSILE:
            color = MISSILE_CONFIGS[m].get("color", missile_colors[m % len(missile_colors)])
        else:
            color = 'red'
        alt_line, = ax.plot([], [], [], color=color, alpha=ALTITUDE_LINE_ALPHA,
                            linestyle=':', linewidth=1)
        altitude_lines.append(alt_line)

# Create velocity vector artists using quiver
velocity_artists = []
if SHOW_VELOCITY_VECTORS:
    # Note: We'll update these in the animation loop
    pass  # Quiver objects will be created dynamically

# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================
def init():
    """Initialize animation."""
    target_point.set_data([], [])
    target_point.set_3d_properties([])
    target_trail.set_data([], [])
    target_trail.set_3d_properties([])

    for m in range(num_missiles):
        missile_points[m].set_data([], [])
        missile_points[m].set_3d_properties([])
        missile_trails[m].set_data([], [])
        missile_trails[m].set_3d_properties([])

    # Initialize distance lines
    if SHOW_DISTANCE_LINES:
        for line in distance_lines:
            line.set_data([], [])
            line.set_3d_properties([])

    # Initialize altitude lines
    if SHOW_ALTITUDE_LINES:
        for line in altitude_lines:
            line.set_data([], [])
            line.set_3d_properties([])

    # Initialize notification texts
    intercept_text.set_text('')
    decoyed_text.set_text('')
    pause_text.set_text('')

    # Reset pause state
    pause_state['paused'] = False

    artists = [target_point, target_trail] + missile_points + missile_trails
    artists += [intercept_text, decoyed_text, pause_text]
    artists += distance_lines + altitude_lines
    return tuple(artists)


def update(frame):
    """Update animation for given frame."""

    # ========== CHECK FOR FREEZE STATE (Jet destroyed) ==========
    jet_destroyed = any(intercepted)

    # If jet is destroyed and we haven't frozen yet, freeze now
    if jet_destroyed and pause_state['frozen_frame'] is None:
        pause_state['frozen_frame'] = frame
        pause_state['paused'] = True
        # Stop the animation
        if pause_state['anim_ref'] is not None:
            pause_state['anim_ref'].event_source.stop()

    # Use frozen frame if jet is destroyed
    display_frame = pause_state['frozen_frame'] if pause_state['frozen_frame'] is not None else frame
    current_time_sec = display_frame * dt

    # ========== UPDATE 3D POSITIONS ==========
    # Update aircraft position
    target_point.set_data([target_states[display_frame, 0]], [target_states[display_frame, 1]])
    target_point.set_3d_properties([target_states[display_frame, 2]])

    # Change Jet marker: CIRCLE (normal) -> STAR (destroyed)
    if jet_destroyed:
        target_point.set_marker('*')  # Star when destroyed
        target_point.set_color('red')
        target_point.set_markersize(20)
    else:
        target_point.set_marker('o')  # Circle when normal
        target_point.set_color('blue')
        target_point.set_markersize(12)

    # Update aircraft trail (with fade effect / max length)
    if TRAIL_FADE_EFFECT:
        trail_start = max(0, display_frame - TRAIL_MAX_LENGTH)
    else:
        trail_start = 0
    target_trail.set_data(target_states[trail_start:display_frame+1, 0], target_states[trail_start:display_frame+1, 1])
    target_trail.set_3d_properties(target_states[trail_start:display_frame+1, 2])

    # Calculate distances for each missile
    missile_distances = []
    for m in range(num_missiles):
        # Update missile position
        missile_points[m].set_data([all_missile_states[m, display_frame, 0]], [all_missile_states[m, display_frame, 1]])
        missile_points[m].set_3d_properties([all_missile_states[m, display_frame, 2]])

        # Update missile trail
        if TRAIL_FADE_EFFECT:
            trail_start = max(0, display_frame - TRAIL_MAX_LENGTH)
        else:
            trail_start = 0
        missile_trails[m].set_data(all_missile_states[m, trail_start:display_frame+1, 0], all_missile_states[m, trail_start:display_frame+1, 1])
        missile_trails[m].set_3d_properties(all_missile_states[m, trail_start:display_frame+1, 2])

        # Calculate distance
        dist = np.linalg.norm(target_states[display_frame] - all_missile_states[m, display_frame])
        missile_distances.append(dist)

        # Update distance lines
        if SHOW_DISTANCE_LINES and m < len(distance_lines):
            m_pos = all_missile_states[m, display_frame]
            t_pos = target_states[display_frame]
            distance_lines[m].set_data([m_pos[0], t_pos[0]], [m_pos[1], t_pos[1]])
            distance_lines[m].set_3d_properties([m_pos[2], t_pos[2]])

    # Update altitude lines
    if SHOW_ALTITUDE_LINES and len(altitude_lines) > 0:
        t_pos = target_states[display_frame]
        altitude_lines[0].set_data([t_pos[0], t_pos[0]], [t_pos[1], t_pos[1]])
        altitude_lines[0].set_3d_properties([0, t_pos[2]])
        for m in range(num_missiles):
            if m + 1 < len(altitude_lines):
                m_pos = all_missile_states[m, display_frame]
                altitude_lines[m + 1].set_data([m_pos[0], m_pos[0]], [m_pos[1], m_pos[1]])
                altitude_lines[m + 1].set_3d_properties([0, m_pos[2]])

    # ========== UPDATE STATUS PANEL ==========
    # JET STATUS
    if jet_destroyed:
        status_texts['jet_status'].set_text('Status: DESTROYED')
        status_texts['jet_status'].set_color('red')
    else:
        status_texts['jet_status'].set_text('Status: ACTIVE')
        status_texts['jet_status'].set_color('#00ff00')

    # FLARE STATUS - use the count at current display frame
    flares_left = flare_counts_per_frame[display_frame]
    status_texts['jet_flare'].set_text(f'Flares: {flares_left}/{FLARE_COUNT}')
    if flares_left == 0:
        status_texts['jet_flare'].set_color('red')
    elif flares_left < FLARE_COUNT:
        status_texts['jet_flare'].set_color('yellow')  # Some used
    else:
        status_texts['jet_flare'].set_color('orange')

    # SAM STATUS UPDATES
    sam_status_texts = [
        (status_texts['sam1_status'], status_texts['sam1_dist'], 0),
        (status_texts['sam2_status'], status_texts['sam2_dist'], 1),
        (status_texts['sam3_status'], status_texts['sam3_dist'], 2),
    ]

    for status_t, dist_t, m in sam_status_texts:
        if m >= num_missiles:
            continue

        dist = missile_distances[m] if m < len(missile_distances) else 0
        dist_t.set_text(f'Distance: {dist:.0f}m')

        if intercepted[m]:
            status_t.set_text('Status: TARGET HIT!')
            status_t.set_color('#00ff00')
        elif m in decoyed_missiles:
            status_t.set_text('Status: HIT FLARE!')
            status_t.set_color('orange')
        elif missile_dead[m]:
            status_t.set_text('Status: LOST')
            status_t.set_color('red')
        elif missile_launched[m]:
            status_t.set_text('Status: INTERCEPTING')
            status_t.set_color('yellow')
        else:
            # Check if this is the next missile to launch (sequential mode)
            prev_failed = True
            if SEQUENTIAL_LAUNCH and m > 0:
                prev_m = m - 1
                prev_failed = (missile_launched[prev_m] and
                              (prev_m in decoyed_missiles or missile_dead[prev_m]) and
                              not intercepted[prev_m])
            if m == 0 or prev_failed:
                status_t.set_text('Status: READY')
                status_t.set_color('yellow')
            else:
                status_t.set_text('Status: STANDBY')
                status_t.set_color('gray')

    # MISSION STATUS
    all_missiles_done = all(
        missile_dead[m] or m in decoyed_missiles or intercepted[m]
        for m in range(num_missiles) if missile_launched[m]
    ) and all(missile_launched)

    if jet_destroyed:
        status_texts['mission_status'].set_text('TARGET DESTROYED!')
        status_texts['mission_status'].set_color('#00ff00')
        status_texts['mission_detail'].set_text('Click NEXT SIMULATION')
    elif all_missiles_done:
        status_texts['mission_status'].set_text('MISSION FAILED')
        status_texts['mission_status'].set_color('red')
        status_texts['mission_detail'].set_text('Aircraft escaped!')
    else:
        status_texts['mission_status'].set_text('IN PROGRESS...')
        status_texts['mission_status'].set_color('white')
        active_count = sum(1 for m in range(num_missiles)
                          if missile_launched[m] and not intercepted[m]
                          and not missile_dead[m] and m not in decoyed_missiles)
        status_texts['mission_detail'].set_text(f'Active missiles: {active_count}')

    # ========== CENTER NOTIFICATIONS ==========
    intercept_msg = ''
    if jet_destroyed:
        # Find which missile hit
        for m in range(num_missiles):
            if intercepted[m]:
                missile_name = MISSILE_CONFIGS[m].get("name", f"SAM-{m+1}")
                intercept_msg = f'TARGET DESTROYED!\n{missile_name} HIT'
                break
    intercept_text.set_text(intercept_msg)

    # Freeze display when jet destroyed
    if jet_destroyed:
        pause_text.set_text('SIMULATION FROZEN\nPress NEXT SIMULATION')
    else:
        pause_text.set_text('')

    # Decoyed notification
    decoyed_msg = ''
    if cm_system is not None:
        for m in range(num_missiles):
            if m in cm_system.decoyed_times:
                decoyed_time = cm_system.decoyed_times[m]
                time_since_decoyed = current_time_sec - decoyed_time
                if 0 <= time_since_decoyed < 3.0:
                    missile_name = MISSILE_CONFIGS[m].get("name", f"SAM-{m+1}")
                    decoyed_msg = f'{missile_name} DECOYED!\nHIT FLARE'
                    break
    decoyed_text.set_text(decoyed_msg)

    artists = [target_point, target_trail] + missile_points + missile_trails
    artists += [intercept_text, decoyed_text, pause_text]
    artists += distance_lines + altitude_lines
    return tuple(artists)


# ============================================================================
# INTERACTIVE GUI CONTROLS
# ============================================================================
if ENABLE_INTERACTIVE_GUI:
    # Adjust figure layout to make room for controls
    plt.subplots_adjust(left=0.05, right=0.75, bottom=0.15, top=0.95)

    # Animation state
    class AnimationState:
        def __init__(self):
            self.is_paused = False
            self.current_frame = 0
            self.speed_multiplier = 1.0
            self.frame_list = None
            self.anim = None

    anim_state = AnimationState()

    # Create control panel axes
    ax_play = plt.axes([0.78, 0.85, 0.08, 0.04])
    ax_pause = plt.axes([0.87, 0.85, 0.08, 0.04])
    ax_reset = plt.axes([0.78, 0.80, 0.17, 0.04])

    ax_speed = plt.axes([0.78, 0.70, 0.17, 0.03])
    ax_time_slider = plt.axes([0.78, 0.60, 0.17, 0.03])
    ax_elev = plt.axes([0.78, 0.50, 0.17, 0.03])
    ax_azim = plt.axes([0.78, 0.40, 0.17, 0.03])

    # Create buttons
    btn_play = Button(ax_play, 'Play', color='lightgreen', hovercolor='green')
    btn_pause = Button(ax_pause, 'Pause', color='lightyellow', hovercolor='yellow')
    btn_reset = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='red')

    # Create sliders
    slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0, valstep=0.1)
    slider_time = Slider(ax_time_slider, 'Time', 0, tmax, valinit=0, valstep=dt*100)
    slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=20, valstep=5)
    slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=45, valstep=5)

    # Info text panel
    ax_info = plt.axes([0.78, 0.15, 0.17, 0.20])
    ax_info.set_facecolor('lightgray')
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.set_title('Simulation Info', fontsize=10)

    info_text = ax_info.text(0.05, 0.95,
        f"Guidance: {GUIDANCE_ALGORITHM}\n"
        f"Physics: {'ON' if ENABLE_PHYSICS else 'OFF'}\n"
        f"Evasion: {EVASION_PATTERN}\n"
        f"Missiles: {num_missiles}\n"
        f"Target vel: {targ_vel} m/s\n"
        f"Missile vel: {miss_vel} m/s",
        transform=ax_info.transAxes, fontsize=8,
        verticalalignment='top', family='monospace')

    # Status indicator
    ax_status = plt.axes([0.78, 0.10, 0.17, 0.04])
    ax_status.set_facecolor('lightgray')
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    status_text = ax_status.text(0.5, 0.5, 'Status: Playing',
        transform=ax_status.transAxes, fontsize=10,
        horizontalalignment='center', verticalalignment='center')

    def on_play(event):
        """Resume animation."""
        anim_state.is_paused = False
        if anim_state.anim is not None:
            anim_state.anim.resume()
        status_text.set_text('Status: Playing')
        fig.canvas.draw_idle()

    def on_pause(event):
        """Pause animation."""
        anim_state.is_paused = True
        if anim_state.anim is not None:
            anim_state.anim.pause()
        status_text.set_text('Status: Paused')
        fig.canvas.draw_idle()

    def on_reset(event):
        """Reset animation to beginning."""
        anim_state.current_frame = 0
        slider_time.set_val(0)
        update(0)
        if anim_state.is_paused:
            status_text.set_text('Status: Reset (Paused)')
        else:
            status_text.set_text('Status: Reset')
        fig.canvas.draw_idle()

    def on_speed_change(val):
        """Change animation speed."""
        anim_state.speed_multiplier = val
        if anim_state.anim is not None:
            anim_state.anim.event_source.interval = animation_interval / val

    def on_time_change(val):
        """Jump to specific time."""
        frame_idx = int(val / tmax * (len(times) - 1))
        frame_idx = max(0, min(frame_idx, len(times) - 1))
        update(frame_idx)
        fig.canvas.draw_idle()

    def on_elev_change(val):
        """Change view elevation."""
        ax.view_init(elev=val, azim=slider_azim.val)
        fig.canvas.draw_idle()

    def on_azim_change(val):
        """Change view azimuth."""
        ax.view_init(elev=slider_elev.val, azim=val)
        fig.canvas.draw_idle()

    # Connect callbacks
    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_reset.on_clicked(on_reset)
    slider_speed.on_changed(on_speed_change)
    slider_time.on_changed(on_time_change)
    slider_elev.on_changed(on_elev_change)
    slider_azim.on_changed(on_azim_change)

    print("Interactive GUI enabled:")
    print("  - Play/Pause/Reset buttons")
    print("  - Speed slider (0.1x - 5x)")
    print("  - Time scrubber")
    print("  - View angle controls")

# ============================================================================
# CREATE AND SHOW ANIMATION
# ============================================================================
# Skip frames for smoother playback (reduce to 500 frames max)
frame_skip = max(1, len(times) // 500)
frames = range(0, len(times), frame_skip)

print(f"Animation will show {len(frames)} frames")

anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                     blit=False, interval=animation_interval, repeat=False)  # No repeat - freeze on destroy

# Store animation reference for freeze control
pause_state['anim_ref'] = anim

# Store animation reference for GUI controls
if ENABLE_INTERACTIVE_GUI:
    anim_state.anim = anim
    anim_state.frame_list = list(frames)

print("Showing animation...")
print("\nControls:")
print("  - Simulation freezes when Jet is destroyed")
print("  - Click NEXT SIMULATION button to run again")
print("  - Drag mouse on 3D plot to rotate view")
plt.show()
