import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
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
# MULTI-MISSILE SETTINGS
# ============================================================================
# Enable multiple missiles
ENABLE_MULTI_MISSILE = True
NUM_MISSILES = 3  # Number of missiles to launch

# Missile launch configuration
# Each missile can have: start_position, launch_time, guidance_algorithm
MISSILE_CONFIGS = [
    {
        "start_pos": np.array([13000, 12000, 0]),
        "launch_time": 0.0,
        "guidance": "proportional_navigation",
        "color": "red"
    },
    {
        "start_pos": np.array([15000, 8000, 2000]),
        "launch_time": 2.0,
        "guidance": "lead_pursuit",
        "color": "orange"
    },
    {
        "start_pos": np.array([10000, 15000, -1000]),
        "launch_time": 4.0,
        "guidance": "augmented_pn",
        "color": "magenta"
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

    # ========== MISSILES UPDATE ==========
    for m in range(num_missiles):
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
            if ENABLE_MULTI_MISSILE:
                print(f"Missile {m+1} launched at t = {t:.2f}s")
            else:
                print(f"Missile launched at t = {t:.2f}s")

        if missile_launched[m]:
            # Missile holds position after intercept or death
            if intercepted[m] or missile_dead[m]:
                all_missile_states[m, i] = all_missile_states[m, i-1]
                all_missile_velocities[m, i] = all_missile_velocities[m, i-1]
                all_missile_speeds[m, i] = all_missile_speeds[m, i-1]
                continue

            # Get target info
            target_pos = target_states[i]
            target_vel = target_velocities[i]
            target_accel = target_accelerations[i]

            # Check for intercept
            distance = np.linalg.norm(target_pos - all_missile_states[m, i-1])
            if distance < kill_dist and intercept_times[m] is None:
                intercept_times[m] = t
                intercept_indices[m] = i
                intercepted[m] = True
                any_intercepted = True
                if ENABLE_MULTI_MISSILE:
                    print(f"Missile {m+1} intercept at t = {t:.2f}s, distance = {distance:.1f}m")
                else:
                    print(f"Intercept at t = {t:.2f}s, distance = {distance:.1f}m")
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
    status = "HIT" if intercepted[m] else ("DEAD" if missile_dead[m] else "MISS")
    if ENABLE_MULTI_MISSILE:
        print(f"Missile {m+1}: {status}, final distance: {final_dist:.1f}m")
    else:
        print(f"Final miss distance: {final_dist:.1f}m")

# ============================================================================
# CREATE 3D PLOT
# ============================================================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

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
if ENABLE_MULTI_MISSILE:
    ax.set_title(f'3D Missile-Aircraft Pursuit Simulation\n{num_missiles} Missiles')
else:
    ax.set_title(f'3D Missile-Aircraft Pursuit Simulation\nGuidance: {GUIDANCE_ALGORITHM.replace("_", " ").title()}')
ax.grid(True)
ax.view_init(elev=20, azim=45)

# Create animation artists
target_point, = ax.plot([], [], [], 'bo', markersize=10, label='Aircraft')
target_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5, label='Aircraft Trail')

# Create artists for each missile
missile_points = []
missile_trails = []
missile_colors = ['red', 'orange', 'magenta', 'cyan', 'yellow', 'lime']

for m in range(num_missiles):
    if ENABLE_MULTI_MISSILE:
        color = MISSILE_CONFIGS[m].get("color", missile_colors[m % len(missile_colors)])
    else:
        color = 'red'
    point, = ax.plot([], [], [], 'o', color=color, markersize=8, label=f'Missile {m+1}' if ENABLE_MULTI_MISSILE else 'Missile')
    trail, = ax.plot([], [], [], '-', color=color, linewidth=1.5, alpha=0.5)
    missile_points.append(point)
    missile_trails.append(trail)

# HUD text
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
speed_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
missile_speed_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10, color='red')
distance_text = ax.text2D(0.02, 0.80, '', transform=ax.transAxes, fontsize=10)
algo_text = ax.text2D(0.02, 0.75, f'Missiles: {num_missiles}' if ENABLE_MULTI_MISSILE else f'Algorithm: {GUIDANCE_ALGORITHM}', transform=ax.transAxes, fontsize=10, color='purple')
physics_text = ax.text2D(0.02, 0.70, f'Physics: {"ON" if ENABLE_PHYSICS else "OFF"}', transform=ax.transAxes, fontsize=10, color='green' if ENABLE_PHYSICS else 'gray')
evasion_text = ax.text2D(0.02, 0.65, f'Evasion: {EVASION_PATTERN if ENABLE_EVASION else "OFF"}', transform=ax.transAxes, fontsize=10, color='orange' if ENABLE_EVASION else 'gray')

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

    time_text.set_text('')
    speed_text.set_text('')
    missile_speed_text.set_text('')
    distance_text.set_text('')

    artists = [target_point, target_trail] + missile_points + missile_trails + [time_text, speed_text, missile_speed_text, distance_text]
    return tuple(artists)


def update(frame):
    """Update animation for given frame."""
    # Update aircraft position
    target_point.set_data([target_states[frame, 0]], [target_states[frame, 1]])
    target_point.set_3d_properties([target_states[frame, 2]])

    # Update aircraft trail
    target_trail.set_data(target_states[:frame+1, 0], target_states[:frame+1, 1])
    target_trail.set_3d_properties(target_states[:frame+1, 2])

    # Update all missiles
    min_distance = float('inf')
    for m in range(num_missiles):
        # Update missile position
        missile_points[m].set_data([all_missile_states[m, frame, 0]], [all_missile_states[m, frame, 1]])
        missile_points[m].set_3d_properties([all_missile_states[m, frame, 2]])

        # Update missile trail
        missile_trails[m].set_data(all_missile_states[m, :frame+1, 0], all_missile_states[m, :frame+1, 1])
        missile_trails[m].set_3d_properties(all_missile_states[m, :frame+1, 2])

        # Track closest missile distance
        dist = np.linalg.norm(target_states[frame] - all_missile_states[m, frame])
        if dist < min_distance:
            min_distance = dist

    # Calculate current target speed (for display)
    if frame > 0:
        dx = target_states[frame, 0] - target_states[frame-1, 0]
        dy = target_states[frame, 1] - target_states[frame-1, 1]
        dz = target_states[frame, 2] - target_states[frame-1, 2]
        target_speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    else:
        target_speed = targ_vel

    # Get average missile speed (or first missile for single mode)
    if ENABLE_MULTI_MISSILE:
        avg_speed = np.mean([all_missile_speeds[m, frame] for m in range(num_missiles)])
        missile_speed_str = f'Avg Missile Speed = {avg_speed:.1f} m/s'
    else:
        missile_speed_str = f'Missile Speed = {all_missile_speeds[0, frame]:.1f} m/s'

    # Update text
    time_text.set_text(f'Time = {times[frame]:.2f} s')
    speed_text.set_text(f'Target Speed = {target_speed:.1f} m/s')
    missile_speed_text.set_text(missile_speed_str)
    distance_text.set_text(f'Closest = {min_distance:.1f} m')

    artists = [target_point, target_trail] + missile_points + missile_trails + [time_text, speed_text, missile_speed_text, distance_text]
    return tuple(artists)


# ============================================================================
# CREATE AND SHOW ANIMATION
# ============================================================================
# Skip frames for smoother playback (reduce to 500 frames max)
frame_skip = max(1, len(times) // 500)
frames = range(0, len(times), frame_skip)

print(f"Animation will show {len(frames)} frames")

anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                     blit=False, interval=animation_interval, repeat=True)

print("Showing animation...")
plt.show()
