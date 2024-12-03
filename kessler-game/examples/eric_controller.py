# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController  # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from kesslergame.asteroid import Asteroid
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

COLLISION_TIME_TOLERANCE = 15
SHIP_RADIUS = 20.0

class EricTestController(KesslerController):

    def __init__(self):
        self.eval_frames = 0  # What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi / 30, math.pi / 30, 0.1),
                                      'theta_delta')  # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        ship_move = ctrl.Consequent(np.arange(-210, 0, 1), 'ship_move')
        ship_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_mine')

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
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        # Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        ship_move['L'] = fuzz.trimf(ship_move.universe, [-150, -125, -75])
        ship_move['M'] = fuzz.trimf(ship_move.universe, [-100, -75, -50])
        ship_move['S'] = fuzz.trimf(ship_move.universe, [-50, -25, 0])
        ship_move['Z'] = fuzz.trimf(ship_move.universe, [0, 0, 0])

        ship_mine['N'] = fuzz.zmf(ship_fire.universe, -1, 0)
        ship_mine['Y'] = fuzz.smf(ship_fire.universe, 0, 1)

        # Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'],
                          (ship_turn['NL'], ship_fire['N'], ship_move['S'], ship_mine['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'],
                          (ship_turn['NM'], ship_fire['N'], ship_move['M'], ship_mine['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'],
                          (ship_turn['NS'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'],
                          (ship_turn['PS'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'],
                          (ship_turn['PM'], ship_fire['N'], ship_move['M'], ship_mine['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'],
                          (ship_turn['PL'], ship_fire['N'], ship_move['S'], ship_mine['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'],
                          (ship_turn['NL'], ship_fire['N'], ship_move['M'], ship_mine['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'],
                          (ship_turn['NM'], ship_fire['N'], ship_move['L'], ship_mine['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'],
                           (ship_turn['NS'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'],
                           (ship_turn['PS'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'],
                           (ship_turn['PM'], ship_fire['N'], ship_move['L'], ship_mine['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'],
                           (ship_turn['PL'], ship_fire['N'], ship_move['M'], ship_mine['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'],
                           (ship_turn['NL'], ship_fire['Y'], ship_move['Z'], ship_mine['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'],
                           (ship_turn['NM'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'],
                           (ship_turn['NS'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'],
                           (ship_turn['PS'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'],
                           (ship_turn['PM'], ship_fire['Y'], ship_move['Z'], ship_mine['N']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'],
                           (ship_turn['PL'], ship_fire['Y'], ship_move['Z'], ship_mine['Y']))

        # Declare the fuzzy controller, add the rules
        # This is an instance variable, and thus available for other methods in the same object. See notes.
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])

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

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        # thrust = 0 <- How do the values scale with asteroid velocity vector?
        # turn_rate = 90 <- How do the values scale with asteroid velocity vector?

        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity.
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
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
            if t_min < 0 or t_min > COLLISION_TIME_TOLERANCE:
                # Asteroid is not on a collision course with the ship for a while
                continue

            d_min = np.linalg.norm(R + V * t_min)
            if d_min < min_d_min:
                min_d_min = d_min
                try:
                    threat_asteroid["aster"] = a
                    threat_asteroid["dist"] = math.sqrt(
                        math.sqrt((ship_pos[0] - a["position"][0]) ** 2 + (ship_pos[1] - a["position"][1]) ** 2)
                    )
                except TypeError:
                    threat_asteroid = {"aster": a, "dist": math.sqrt(
                        math.sqrt((ship_pos[0] - a["position"][0]) ** 2 + (ship_pos[1] - a["position"][1]) ** 2)
                    )}


        ship_vel = ship_state["velocity"]
        ship_pos = ship_state["position"]
        ship_r = SHIP_RADIUS
        min_collision_time = float("inf")

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

            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-B + sqrt_discriminant) / (2 * A)
            t2 = (-B - sqrt_discriminant) / (2 * A)

            possible_collision_times = [t for t in [t1, t2] if t >= 0]
            if not possible_collision_times:
                # No collision possible
                continue
            collision_time = min(possible_collision_times)

            if collision_time > COLLISION_TIME_TOLERANCE:
                # No collision possible
                continue

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

        ship_pos_x = ship_state["position"][0]  # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        for a in game_state["asteroids"]:
            # Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0]) ** 2 + (ship_pos_y - a["position"][1]) ** 2)
            if closest_asteroid is None:
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster=a, dist=curr_dist)

            else:
                # closest_asteroid exists, and is thus initialized.
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        if threat_asteroid is None:
            threat_asteroid = {"aster": closest_asteroid["aster"], "dist": closest_asteroid["dist"]}


        # closest_asteroid is now the nearest asteroid object.
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.

        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!

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

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
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

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi / 180) * ship_state["heading"])



        # calculating mine distance
        closest_mine = None
        for mine in game_state["mines"]:
            curr_dist = math.sqrt((ship_pos_x - mine["position"][0]) ** 2 + (ship_pos_y - mine["position"][1]) ** 2)
            if closest_mine is None:
                closest_mine = dict(m=mine, dist=curr_dist)

            else:
                if closest_mine["dist"] > curr_dist:
                    closest_mine["m"] = mine
                    closest_mine["dist"] = curr_dist

                # still in testing, replaces the closest asteroid with a mine
                # if the mine is more threatening.
                # basically going to use these two distances as our highest threat value targets
                elif closest_asteroid["dist"] > curr_dist:
                    closest_asteroid["aster"] = mine
                    closest_asteroid["dist"] = curr_dist

        # if the mine exists and is within 100 meters
        # 100 meters was arbitrarily chosen, setting it to mine radius + ship radius was too small
        if closest_mine is not None and abs(closest_mine["dist"]) < 200.0:
            # was going to use this for fuzzy input but doesn't work very well
            mine_threat = 100

            # calculating the midpoint of the two largest threats
            # asteroid threat is currently just distance based
            mine_aster_midpoint_y = (closest_mine["m"]["position"][1] +
                                     + closest_asteroid["aster"]["position"][1]) / 2
            mine_aster_midpoint_x = (closest_mine["m"]["position"][0]
                                     + closest_asteroid["aster"]["position"][0]) / 2

            my_theta3 = math.atan2((mine_aster_midpoint_y - ship_pos_y), (mine_aster_midpoint_x - ship_pos_x))

            # replace the shooting angle so we always run away from both the mine and the highest threat asteroid
            shooting_theta = my_theta3 - ((math.pi / 180) * ship_state["heading"])

        else:
            mine_threat = 0


        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        shooting.compute()

        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']

        ship_vel_vector = math.sqrt(ship_vel[0] ** 2 + ship_vel[1] ** 2)

        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        if mine_threat < 100 and not fire:
            if abs(shooting_theta) > math.pi / 2:  # Asteroid is behind the ship
                thrust = - shooting.output['ship_move']
            else:
                thrust = shooting.output['ship_move']
        elif mine_threat == 100:
            # we want to run away from the mine
            thrust = -250 + abs(ship_vel_vector)
        else:
            if ship_vel_vector > 1.0:
                thrust = -1 * ship_vel_vector / abs(ship_vel_vector)
            else:
                thrust = 0

        if shooting.output['ship_mine'] >= 0.5:
            drop_mine = True
        else:
            drop_mine = False

        self.eval_frames += 1

        # DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire, shooting.output['ship_mine'])

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "EricTestController"