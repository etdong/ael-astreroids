# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController  # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

SHIP_RADIUS = 20.0

class AEL_CompetitionController(KesslerController):

    def __init__(self):
        self.eval_frames = 0  # What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'bullet_time')  # Updated as per class email
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi / 30, math.pi / 30, 0.1),
                                      'theta_delta')  # Radians due to Python
        collision_time = ctrl.Antecedent(np.arange(0, 5, 0.1), 'collision_time')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        thrust = ctrl.Consequent(np.arange(0.0, 240.0, 0.1), 'thrust') # Anything beyond 120 is gets tough to control

        # Declare fuzzy sets for collision_time (time to collision with the nearest asteroid)
        collision_time['S'] = fuzz.trimf(collision_time.universe, [0, 0, 1])
        collision_time['M'] = fuzz.trimf(collision_time.universe, [1, 3, 5])
        collision_time['L'] = fuzz.trimf(collision_time.universe, [4, 10, 10])

        # Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 0.1)

        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1 * math.pi / 30, -2 * math.pi / 90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1 * math.pi / 30, -2 * math.pi / 90, -1 * math.pi / 90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2 * math.pi / 90, -1 * math.pi / 90, math.pi / 90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1 * math.pi / 90, math.pi / 90, 2 * math.pi / 90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi / 90, 2 * math.pi / 90, math.pi / 30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, 2 * math.pi / 90, math.pi / 30)

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60, 0, 60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        # Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be
        # thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        # Declare fuzzy sets for thrust; this will be returned as thrust.
        # thrust['Z'] = fuzz.trimf(thrust.universe, [-240.0, 0.0, 240.0])
        thrust['PS'] = fuzz.trimf(thrust.universe, [0.0, 0.0, 120.0])
        thrust['PM'] = fuzz.trimf(thrust.universe, [60.0, 120.0, 180.0])
        thrust['PL'] = fuzz.trimf(thrust.universe, [120.0, 240.0, 240.0])

        # Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        # DEBUG
        # bullet_time.view()
        # theta_delta.view()
        # ship_turn.view()
        # ship_fire.view()

        # Declare the fuzzy controller, add the rules
        # This is an instance variable, and thus available for other methods in the same object. See notes.
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
        # rule10, rule11, rule12, rule13, rule14, rule15])

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)

        # Declare each fuzzy rule
        mc_rule1 = ctrl.Rule(collision_time['L'] & theta_delta['NL'], (ship_turn['NL'], thrust['PS']))
        mc_rule2 = ctrl.Rule(collision_time['L'] & theta_delta['NM'], (ship_turn['NM'], thrust['PS']))
        mc_rule3 = ctrl.Rule(collision_time['L'] & theta_delta['NS'], (ship_turn['NS'], thrust['PM']))
        mc_rule5 = ctrl.Rule(collision_time['L'] & theta_delta['PS'], (ship_turn['PS'], thrust['PM']))
        mc_rule6 = ctrl.Rule(collision_time['L'] & theta_delta['PM'], (ship_turn['PM'], thrust['PS']))
        mc_rule7 = ctrl.Rule(collision_time['L'] & theta_delta['PL'], (ship_turn['PL'], thrust['PS']))
        mc_rule8 = ctrl.Rule(collision_time['M'] & theta_delta['NL'], (ship_turn['NL'], thrust['PS']))
        mc_rule9 = ctrl.Rule(collision_time['M'] & theta_delta['NM'], (ship_turn['NM'], thrust['PM']))
        mc_rule10 = ctrl.Rule(collision_time['M'] & theta_delta['NS'], (ship_turn['NS'], thrust['PM']))
        mc_rule12 = ctrl.Rule(collision_time['M'] & theta_delta['PS'], (ship_turn['PS'], thrust['PM']))
        mc_rule13 = ctrl.Rule(collision_time['M'] & theta_delta['PM'], (ship_turn['PM'], thrust['PM']))
        mc_rule14 = ctrl.Rule(collision_time['M'] & theta_delta['PL'], (ship_turn['PL'], thrust['PS']))
        mc_rule15 = ctrl.Rule(collision_time['S'] & theta_delta['NL'], (ship_turn['NL'], thrust['PM']))
        mc_rule16 = ctrl.Rule(collision_time['S'] & theta_delta['NM'], (ship_turn['NM'], thrust['PM']))
        mc_rule17 = ctrl.Rule(collision_time['S'] & theta_delta['NS'], (ship_turn['NS'], thrust['PL']))
        mc_rule19 = ctrl.Rule(collision_time['S'] & theta_delta['PS'], (ship_turn['PS'], thrust['PL']))
        mc_rule20 = ctrl.Rule(collision_time['S'] & theta_delta['PM'], (ship_turn['PM'], thrust['PM']))
        mc_rule21 = ctrl.Rule(collision_time['S'] & theta_delta['PL'], (ship_turn['PL'], thrust['PM']))
        self.moving_control = ctrl.ControlSystem()
        self.moving_control.addrule(mc_rule1)
        self.moving_control.addrule(mc_rule2)
        self.moving_control.addrule(mc_rule3)
        self.moving_control.addrule(mc_rule5)
        self.moving_control.addrule(mc_rule6)
        self.moving_control.addrule(mc_rule7)
        self.moving_control.addrule(mc_rule8)
        self.moving_control.addrule(mc_rule9)
        self.moving_control.addrule(mc_rule10)
        self.moving_control.addrule(mc_rule12)
        self.moving_control.addrule(mc_rule13)
        self.moving_control.addrule(mc_rule14)
        self.moving_control.addrule(mc_rule15)
        self.moving_control.addrule(mc_rule16)
        self.moving_control.addrule(mc_rule17)
        self.moving_control.addrule(mc_rule19)
        self.moving_control.addrule(mc_rule20)
        self.moving_control.addrule(mc_rule21)


    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        # thrust = 0 <- How do the values scale with asteroid velocity vector?
        # turn_rate = 90 <- How do the values scale with asteroid velocity vector?

        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity.
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded
        # in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.


        ship_vel = ship_state["velocity"]
        ship_pos = ship_state["position"]
        min_d_min = float("inf")
        threat_asteroid = None

        for a in game_state["asteroids"]:
            ast_pos = a["position"]
            ast_vel = a["velocity"]

            R = np.array(ast_pos) - np.array(ship_pos)
            V = np.array(ast_vel) - np.array(ship_vel)

            V_dot_V = np.dot(V, V)

            if V_dot_V == 0:
                # Asteroid is stationary relative to the ship
                continue
            t_min = -np.dot(R, V) / V_dot_V
            if t_min < 0:
                # Asteroid is not on a collision course with the ship for a while
                continue

            d_min = np.linalg.norm(R + V * t_min)
            if d_min < min_d_min:
                min_d_min = d_min
                try:
                    threat_asteroid["aster"] = a
                    threat_asteroid["dist"] = math.sqrt((ship_pos[0] - a["position"][0]) ** 2 + (ship_pos[1] -
                                                                                                 a["position"][1]) ** 2)
                except TypeError:
                    threat_asteroid = {"aster": a, "dist": math.sqrt((ship_pos[0] - a["position"][0]) ** 2 + (ship_pos[1] - a["position"][1]) ** 2)}

        ship_vel = ship_state["velocity"]
        ship_pos = ship_state["position"]
        ship_r = SHIP_RADIUS
        min_collision_time = float("inf")
        danger_level = 0

        for a in game_state["asteroids"]:
            ast_pos = a["position"]
            ast_vel = a["velocity"]
            ast_r = a["size"] * 8.0

            R = np.array(ast_pos) - np.array(ship_pos)
            V = np.array(ast_vel) - np.array(ship_vel)

            collsion_distance = ast_r + ship_r
            collsion_distance_squared = collsion_distance ** 2

            # coefficients of the quadratic equation
            A = np.dot(V, V)
            B = 2 * np.dot(R, V)
            C = np.dot(R, R) - collsion_distance_squared

            # discriminant of the quadratic equation
            discriminant = B ** 2 - 4 * A * C

            if discriminant < 0 or A == 0:
                # No collision possible
                continue

            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-B + sqrt_discriminant) / (2 * A)
            t2 = (-B - sqrt_discriminant) / (2 * A)

            possible_collision_times = [t for t in [t1, t2] if t >= 0]
            if not possible_collision_times:
                # No collision possible
                continue
            collision_time = min(possible_collision_times)

            """At this point we know that a collision is possible so adjust danger level
            based on collision time and asteroid size note that asteroid of size 4 spawns
             3 asteroids of size 3, so on. The danger level is calculated as the total number of size 1 
             asteroids divided by the collision time. Once we aggregate over all asteroids we can then
             determine if the ships fire rate can even keep up with the number of asteroids that will be. If it can't
             we flee."""
            num_bullets_required = (3 ** (a["size"]) - 1)/2
            danger_level += num_bullets_required / collision_time

            if collision_time < min_collision_time:
                min_collision_time = collision_time
                try:
                    threat_asteroid["aster"] = a
                    threat_asteroid["dist"] = math.sqrt(
                        math.sqrt((ship_pos[0] - a["position"][0]) ** 2 + (ship_pos[1] - a["position"][1]) ** 2)
                    )
                except TypeError:
                    threat_asteroid = {"aster": a, "dist": math.sqrt(
                        math.sqrt((ship_pos[0] - a["position"][0]) ** 2 + (ship_pos[1] - a["position"][1]) ** 2)
                    )}

        # threat asteroid is now our target

        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!

        if threat_asteroid is None:  # just in case
            return 0, 0, False, False

        if danger_level/2 > ship_state["fire_rate"]:
            """
            Danger level is scaled back for two reasons
                - Bullet impulse actually slows down the approaching asteroid
                - The ship rarely needs to shoot all the asteroids to avoid collision
                  since child asteroids spread out from parents trajectory
            """
            print("Boutta die!")

            ship_x, ship_y = ship_state["position"]

            num_directions = 90 # This could totally screw up performance - might need to reduce
            collision_threshold = 50.0  # idk
            best_direction = None
            best_length = -1.0

            # Generate directions evenly spaced around the circle
            for i in range(num_directions):
                angle = (2 * math.pi / num_directions) * i
                direction_vec = np.array([math.cos(angle), math.sin(angle)])

                # For this direction, find how far we can go before collision
                # Start with a very large safe length
                safe_length = float('inf')

                for a in game_state["asteroids"]:
                    ast_pos = np.array(a["position"])

                    # Project asteroid position onto the direction vector starting at ship
                    # Relative position of asteroid from ship
                    rel_pos = ast_pos - np.array([ship_x, ship_y])
                    # Project rel_pos onto direction_vec to find how far along direction line the asteroid is
                    forward_dist = np.dot(rel_pos, direction_vec)

                    # If forward_dist is negative, asteroid is behind us in this direction, so no threat from that asteroid
                    if forward_dist < 0:
                        continue

                    # Closest point online in direction_vec to asteroid is (ship_pos + direction_vec * forward_dist)
                    closest_point = np.array([ship_x, ship_y]) + direction_vec * forward_dist
                    dist_to_line = np.linalg.norm(ast_pos - closest_point)

                    # If the asteroid comes within collision_threshold at a certain forward_dist,
                    # that means we cannot safely travel beyond (forward_dist - margin).
                    # We'll use forward_dist as a limit if dist_to_line < collision_threshold.
                    if dist_to_line < collision_threshold:
                        # The ship hits the "danger zone" of this asteroid at approximately forward_dist
                        # Reduce safe_length if this asteroid is more limiting
                        if forward_dist < safe_length:
                            safe_length = forward_dist

                # After checking all asteroids, safe_length is how far we can go in this direction
                if safe_length > best_length:
                    best_length = safe_length
                    best_direction = direction_vec

            # Now we have the best direction to go in
            # We'll turn towards that direction
            ship_heading_rad = (math.pi / 180) * ship_state["heading"]

            safe_angle = math.atan2(best_direction[1], best_direction[0])
            delta_theta = safe_angle - ship_heading_rad
            delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi

            moving = ctrl.ControlSystemSimulation(self.moving_control, flush_after_run=1)
            moving.input['collision_time'] = min_collision_time
            moving.input['theta_delta'] = delta_theta

            moving.compute()

            turn_rate = moving.output['ship_turn']
            thrust = moving.output['thrust']
            fire = False
            drop_mine = False
            return thrust, turn_rate, fire, drop_mine

        asteroid_ship_x = ship_pos[0] - threat_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos[1] - threat_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)

        asteroid_direction = math.atan2(threat_asteroid["aster"]["velocity"][1], threat_asteroid["aster"]["velocity"][
            0])  # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(
            threat_asteroid["aster"]["velocity"][0] ** 2 + threat_asteroid["aster"]["velocity"][1] ** 2)
        bullet_speed = 800  # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * threat_asteroid["dist"] * asteroid_vel * cos_my_theta2) ** 2 - (
                    4 * (asteroid_vel ** 2 - bullet_speed ** 2) * (threat_asteroid["dist"] ** 2))

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are
        # two values produced.
        intrcpt1 = ((2 * threat_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (
                    2 * (asteroid_vel ** 2 - bullet_speed ** 2))
        intrcpt2 = ((2 * threat_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (
                    2 * (asteroid_vel ** 2 - bullet_speed ** 2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = threat_asteroid["aster"]["position"][0] + threat_asteroid["aster"]["velocity"][0] * (
                    bullet_t + 1 / 30)
        intrcpt_y = threat_asteroid["aster"]["position"][1] + threat_asteroid["aster"]["velocity"][1] * (
                    bullet_t + 1 / 30)

        my_theta1 = math.atan2((intrcpt_y - ship_pos[1]), (intrcpt_x - ship_pos[0]))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation.
        # BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi / 180) * ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        shooting.compute()

        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']

        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        drop_mine = False

        self.eval_frames += 1

        thrust = 0

        # DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "AEL Competition Controller"
