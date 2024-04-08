import sys
import time
from functools import partial  # pip install functools
import copy
import random

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)
def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        
        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if( u.get_cargo_space_left() >= resource_amount and 
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
                                    # Change targets, because neither target can accept all our resources and 
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u
    
    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()
    
    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)

########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART), # Transfer to nearby cart
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER), # Transfer to nearby worker
            SpawnCityAction,
            PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        self.observation_shape = (3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3 + 3,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}
        self.previous_city_tile_count = 0

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        observation_index = 0
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.

            # Build a list of object nodes by type for quick distance-searches
            self.object_nodes = {}

            # Add resources
            for cell in game.map.resources:
                if cell.resource.type not in self.object_nodes:
                    self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
                else:
                    self.object_nodes[cell.resource.type] = np.concatenate(
                        (
                            self.object_nodes[cell.resource.type],
                            [[cell.pos.x, cell.pos.y]]
                        ),
                        axis=0
                    )

            # Add your own and opponent units
            for t in [team, (team + 1) % 2]:
                for u in game.state["teamStates"][team]["units"].values():
                    key = str(u.type)
                    if t != team:
                        key = str(u.type) + "_opponent"

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[u.pos.x, u.pos.y]]
                            )
                            , axis=0
                        )

            # Add your own and opponent cities
            for city in game.cities.values():
                for cells in city.city_cells:
                    key = "city"
                    if city.team != team:
                        key = "city_opponent"

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[cells.pos.x, cells.pos.y]]
                            )
                            , axis=0
                        )

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs = np.zeros(self.observation_shape)
        
        # Update the type of this object
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        observation_index = 0
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                obs[observation_index] = 1.0 # Worker
            else:
                obs[observation_index+1] = 1.0 # Cart
        if city_tile is not None:
            obs[observation_index+2] = 1.0 # CityTile
        observation_index += 3
        
        pos = None
        if unit is not None:
            pos = unit.pos
        else:
            pos = city_tile.pos

        if pos is None:
            observation_index += 7 * 5 * 2
        else:
            # Encode the direction to the nearest objects
            #   5x direction_nearest
            #   1x distance
            for distance_function in [closest_node, furthest_node]:
                for key in [
                    Constants.RESOURCE_TYPES.WOOD,
                    Constants.RESOURCE_TYPES.COAL,
                    Constants.RESOURCE_TYPES.URANIUM,
                    "city",
                    str(Constants.UNIT_TYPES.WORKER)]:
                    # Process the direction to and distance to this object type

                    # Encode the direction to the nearest object (excluding itself)
                    #   5x direction
                    #   1x distance
                    if key in self.object_nodes:
                        if (    # if city   OR   unit (of obs) == key (of array)
                                (key == "city" and city_tile is not None) or        
                                (unit is not None and str(unit.type) == key and len(game.map.get_cell_by_pos(unit.pos).units) <= 1 )
                        ):
                            # Filter out the current unit from the closest-search
                            closest_index = closest_node((pos.x, pos.y), self.object_nodes[key])
                            filtered_nodes = np.delete(self.object_nodes[key], closest_index, axis=0)
                        else:
                            filtered_nodes = self.object_nodes[key]

                        if len(filtered_nodes) == 0:
                            # No other object of this type
                            obs[observation_index + 5] = 1.0
                        else:
                            # There is another object of this type
                            closest_index = distance_function((pos.x, pos.y), filtered_nodes)

                            if closest_index is not None and closest_index >= 0:
                                closest = filtered_nodes[closest_index]
                                closest_position = Position(closest[0], closest[1])
                                direction = pos.direction_to(closest_position)
                                mapping = {
                                    Constants.DIRECTIONS.CENTER: 0,
                                    Constants.DIRECTIONS.NORTH: 1,
                                    Constants.DIRECTIONS.WEST: 2,
                                    Constants.DIRECTIONS.SOUTH: 3,
                                    Constants.DIRECTIONS.EAST: 4,
                                }
                                obs[observation_index + mapping[direction]] = 1.0  # One-hot encoding direction

                                # 0 to 1 distance
                                distance = pos.distance_to(closest_position)
                                obs[observation_index + 5] = min(distance / 20.0, 1.0)

                                # 0 to 1 value (amount of resource, cargo for unit, or fuel for city)
                                if key == "city":
                                    # City fuel as % of upkeep for 200 turns
                                    c = game.cities[game.map.get_cell_by_pos(closest_position).city_tile.city_id]
                                    obs[observation_index + 6] = min(
                                        c.fuel / (c.get_light_upkeep() * 200.0),
                                        1.0
                                    )
                                elif key in [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL,
                                             Constants.RESOURCE_TYPES.URANIUM]:
                                    # Resource amount
                                    obs[observation_index + 6] = min(
                                        game.map.get_cell_by_pos(closest_position).resource.amount / 500,
                                        1.0
                                    )
                                else:
                                    # Unit cargo
                                    obs[observation_index + 6] = min(
                                        next(iter(game.map.get_cell_by_pos(
                                            closest_position).units.values())).get_cargo_space_left() / 100,
                                        1.0
                                    )

                    observation_index += 7



        if unit is not None:
            # Encode the cargo space
            #   1x cargo size
            obs[observation_index] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][
                "WORKER"]
            observation_index += 1
        else:
            observation_index += 1

        # Game state observations

        #   1x is night
        obs[observation_index] = game.is_night()
        observation_index += 1

        #   1x percent of game done
        obs[observation_index] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_index += 1

        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        max_count = 30
        for key in ["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                obs[observation_index] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                obs[observation_index + 1] = len(self.object_nodes[(key + "_opponent")]) / max_count
            observation_index += 2

        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs[observation_index] = game.state["teamStates"][team]["researchPoints"] / 200.0
        obs[observation_index+1] = float(game.state["teamStates"][team]["researched"]["coal"])
        obs[observation_index+2] = float(game.state["teamStates"][team]["researched"]["uranium"])
        observation_index += 3

        ################### New OBS ###################
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                obs[observation_index] = unit.cooldown

                if (unit.get_cargo_space_left() > 1):
                    obs[observation_index + 1] = 1
                else:
                    obs[observation_index + 2] = 1
        observation_index += 3

        


        # f = open("output.txt", "a")
        # print("OBS: ", obs, file=f)
        return obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            ## If city_tile: choose spawn/research action
            if city_tile != None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            ## If NOT city_tile: choose movement actions (center, north, etc)
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    
                    team=team,
                    x=x,
                    y=y
                )
            
            # if hasattr(action, 'direction'):
            #     print("actionTaken: MoveAction - ", action.direction)
            # else:
            #     print("actionTaken: ", action)
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0

        # new variables
        self.num_of_wood_sources_last = 0
        self.city_count_near_resources = 0

    def get_end_game_modifier(turn, threshold=280, start_modifier=1.0, end_modifier=1.5, max_turns=360):
        if turn <= threshold:
            return start_modifier
        elif turn >= max_turns:
            return end_modifier
        else:
            # Linear interpolation between start_modifier and end_modifier
            return start_modifier + (end_modifier - start_modifier) * ((turn - threshold) / (max_turns - threshold))


    # def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
    #     """
    #     Returns the reward function for this step of the game. Reward should be a
    #     delta increment to the reward, not the total current reward.
    #     """
    #     ############# new variables ##################
    #     teamState = game.state["teamStates"][self.team]

    #     end_game_neg_modifier = 1
    #     end_game_pos_modifier = 1
    #     if (game.state["turn"] > 280):
    #         end_game_neg_modifier = -1.5
    #         end_game_pos_modifier = 1.5
    #     ##############################################


    #     if is_game_error:
    #         # Game environment step failed, assign a game lost reward to not incentivise this
    #         print("Game failed due to error")
    #         return -1.0

    #     if not is_new_turn and not is_game_finished:
    #         # Only apply rewards at the start of each turn or at game end
    #         return 0

    #     # Get some basic stats
    #     unit_count = len(game.state["teamStates"][self.team]["units"])

    #     city_count = 0
    #     city_count_opponent = 0
    #     city_tile_count = 0
    #     city_tile_count_opponent = 0
    #     for city in game.cities.values():
    #         if city.team == self.team:
    #             city_count += 1
    #         else:
    #             city_count_opponent += 1

    #         for cell in city.city_cells:
    #             if city.team == self.team:
    #                 city_tile_count += 1
    #             else:
    #                 city_tile_count_opponent += 1
        
    #     rewards = {}
        
    #     # Give a reward for unit creation/death. 0.05 reward per unit.
    #     rewards["rew/r_units"] = (unit_count - self.units_last) * 0.1
    #     self.units_last = unit_count

    #     # Give a reward for city creation/death. 0.1 reward per city.
    #     rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1 * end_game_pos_modifier
    #     self.city_tiles_last = city_tile_count

    #     # Reward collecting fuel and divided by 20000
    #     fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
    #     rewards["rew/r_fuel_collected"] = ( (fuel_collected - self.fuel_collected_last) / 20000 )
    #     self.fuel_collected_last = fuel_collected 
        
    #     ############################# ADD MORE REWARDS ########################
    #     f = open("output.txt", "a")

    #     ## for each wood that is still alive, give it 1.0 reward (promote agent to mine other wood)
    #     if Constants.RESOURCE_TYPES.WOOD in game.map.resources_by_type:
    #         num_of_wood_sources = len(game.map.resources_by_type[Constants.RESOURCE_TYPES.WOOD])
    #         rewards["rew/d_wood_alive"] = (num_of_wood_sources - self.num_of_wood_sources_last) * 0.4 * end_game_neg_modifier
    #         self.num_of_wood_sources_last = num_of_wood_sources

    #     # slightly more focus on collecting more coal or uranium
    #     rewards["rew/r_coal_mined"] = game.stats["teamStats"][self.team]["resourcesCollected"]["coal"] * 0.02
    #     rewards["rew/r_uranium_mined"] = game.stats["teamStats"][self.team]["resourcesCollected"]["uranium"] * 0.03

    #     ### Research Reward System ###
    #     # rewards["rew/research_point"] = 0
    #     # limit = 250
    #     # if (teamState["researched"]["coal"] == True):
    #     #     limit -= 50
    #     # if (teamState["researched"]["uranium"] == True):
    #     #     limit -= 200
    #     # if (teamState["researchPoints"] <= limit):
    #     #     rewards["rew/research_point"] += teamState["researchPoints"] / 10
    #     # else:
    #     #     rewards["rew/research_point"] -= (teamState["researchPoints"] - limit) / 10
    #     # rewards["rew/research_point"] *= 0.1

    #     # Count of CityTiles near a resource
    #     count = 0
    #     for key in game.map.resources_by_type:
    #         for cell in game.map.resources_by_type[key]:
    #             for adjCell in game.map.get_adjacent_cells_with_corners(cell):
    #                 if adjCell.is_city_tile():
    #                     count += 1
    #     rewards["rew/close_to_resources"] = (count - self.city_count_near_resources) * 0.2
    #     self.city_count_near_resources = count
                

    #     ######################################################################


    #     # Give a reward of 1.0 per city tile alive at the end of the game
    #     rewards["rew/r_city_tiles_end"] = 0
    #     if is_game_finished:
    #         self.is_last_turn = True
    #         rewards["rew/r_city_tiles_end"] = city_tile_count

    #         '''
    #         # Example of a game win/loss reward instead
    #         if game.get_winning_team() == self.team:
    #             rewards["rew/r_game_win"] = 100.0 # Win
    #         else:
    #             rewards["rew/r_game_win"] = -100.0 # Loss
    #         '''

    #         totalResearchReward = 0
    #         if (teamState["researched"]["coal"] == True):
    #             totalResearchReward += 3
    #         else:
    #             totalResearchReward -= 1
    #         if (teamState["researched"]["uranium"] == True):
    #             totalResearchReward += 10
    #         else:
    #             totalResearchReward -= 1
    #         rewards["rew/research_reward"] = totalResearchReward
            
        
    #     reward = 0
    #     for name, value in rewards.items():
    #         reward += value

    #     return reward

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game, focusing on maximizing city building towards the end.
        """

        if is_game_error:
            # Penalize game errors
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        current_turn = game.state["turn"]
        total_turns = GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        threshold_turn = 280  # Start focusing on city building past this turn

        # Calculate dynamic modifiers
        late_game_phase = current_turn > threshold_turn
        late_game_multiplier = 2.0 if late_game_phase else 1.0  # Encourage city building in the late game
        resource_penalty_modifier = (current_turn - threshold_turn) / (total_turns - threshold_turn) if late_game_phase else 0

        # Initialize rewards dictionary
        rewards = {}

        # Basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])
        city_tile_count = sum(len(city.city_cells) for city in game.cities.values() if city.team == self.team)

        # Unit creation/death reward
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05 * (0.5 if late_game_phase else 1.0)
        self.units_last = unit_count

        # City creation reward, dynamically scaled
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1 * late_game_multiplier
        self.city_tiles_last = city_tile_count

        # Fuel collected reward, adjusted for late game
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = (fuel_collected - self.fuel_collected_last) / 20000
        self.fuel_collected_last = fuel_collected

        # Adjust rewards for resource collection, especially in the late game
        for resource_type, modifier in [("coal", 0.02), ("uranium", 0.03)]:
            amount_collected = game.stats["teamStats"][self.team]["resourcesCollected"][resource_type]
            rewards[f"rew/r_{resource_type}_mined"] = amount_collected * modifier #* max(0, 1 - resource_penalty_modifier)

        # Reward for city tiles near resources, to promote strategic city placement
        ## ** NEW REWARD FUNCTION ** ##
        rewards["rew/close_to_resources"] = self.calculate_cities_near_resources(game) * 0.15

        ## ** NEW REWARD FUNCTION ** ##
        rewards["rew/fuel_efficiency"] = self.reward_for_fuel_efficiency(game)

        # Reward for surviving the night
        rewards["rew/city_survival"] = self.reward_for_city_survival(game)

        # End-of-game rewards
        if is_game_finished:
            # Reward based on city tiles at the end
            # rewards["rew/r_city_tiles_end"] = city_tile_count * late_game_multiplier # new
            rewards["rew/r_city_tiles_end"] = city_tile_count * 5 # new

            # Additional rewards or penalties based on research and game outcome could be added here

        # Aggregate reward
        reward = sum(rewards.values())
        return reward

    ## ********* NEW REWARD FUNCTION (WORKS) ********* ##               ((PPO_2 + PPO_4 + PPO_5))
    def calculate_cities_near_resources(self, game):
        """
        Calculate the number of city tiles that are near resources for additional strategic placement rewards.
        """
        count = 0
        for resource_cells in game.map.resources_by_type.values():
            for resource_cell in resource_cells:
                for adj_cell in game.map.get_adjacent_cells_with_corners(resource_cell):
                    if adj_cell.is_city_tile() and adj_cell.city_tile.team == self.team:
                        count += 1
        return count
    ## ********* NEW REWARD FUNCTION ********* ##

    ## ********* NEW REWARD FUNCTION (TESTING) ********* ##                  ((PPO_3))
    # def calculate_clustered_cities_near_resources(self, game):
    #     """
    #     Calculate the number of city tiles that are near resources, with an additional reward for being 
    #     adjacent to other city tiles. This encourages building clusters of cities near resources.
        
    #     Returns:
    #     - count: A weighted count that is higher when city tiles are clustered together near resources.
    #     """
    #     count = 0
    #     clustered_bonus = 0.5  # Additional weight for each adjacent friendly city tile
    #     for resource_type, resource_cells in game.map.resources_by_type.items():
    #         for resource_cell in resource_cells:
    #             # Check all cells adjacent to resources including diagonals
    #             for adj_cell in game.map.get_adjacent_cells_with_corners(resource_cell):
    #                 if adj_cell.is_city_tile() and adj_cell.city_tile.team == self.team:
    #                     # Direct count for a city tile next to a resource
    #                     count += 1
    #                     # Check the number of adjacent friendly city tiles to this city tile
    #                     adjacent_city_tiles = game.map.get_adjacent_cells(adj_cell)
    #                     for adjacent_city_tile in adjacent_city_tiles:
    #                         if (adjacent_city_tile.is_city_tile() and
    #                                 adjacent_city_tile.city_tile.team == self.team):
    #                             # Increment the count with a bonus for clustering
    #                             count += clustered_bonus
    #     return count
    ## ********* NEW REWARD FUNCTION (TESTING) ********* ##
    
    ## ********* NEW REWARD FUNCTION (TESTING) ********* ##                 ((PPO_5))
    def reward_for_fuel_efficiency(self, game):
        night_turns = 10  # last 10 turns in a cycle are night turns
        fuel_efficiency_reward = 0.0
        reward_for_worker_inside = 0.05  # reward for each worker inside a city at night
        reward_for_city_prepared_for_night = 1  # reward for each city tile with enough fuel to survive the night
        
        # Check if it's night time
        if game.is_night():
            # Reward keeping workers inside cities at night
            for worker in game.state["teamStates"][self.team]["units"].values():
                if worker.type == "worker" and worker.is_in_city():
                    fuel_efficiency_reward += reward_for_worker_inside
            
            # # Reward city tiles for having enough fuel to survive the night       ((PPO_4))
            # for city in game.cities.values():
            #     if city.team == self.team:
            #         for city_tile in city.city_cells:
            #             # Calculate the expected fuel consumption for the night
            #             expected_fuel_consumption = 0
            #             for _ in range(night_turns):
            #                 # Get the number of adjacent friendly city tiles to reduce fuel burn
            #                 num_adjacent = sum(1 for adj_cell in game.map.get_adjacent_cells(city_tile) 
            #                                 if adj_cell.city_tile and adj_cell.city_tile.team == self.team)
            #                 expected_fuel_consumption += max(0, 23 - 5 * num_adjacent)
                        
            #             # Check if the city has enough fuel to survive the night
            #             if city.fuel > expected_fuel_consumption:
            #                 fuel_efficiency_reward += reward_for_city_prepared_for_night

        return fuel_efficiency_reward
    ## ********* NEW REWARD FUNCTION (TESTING)********* ##

    ## ********* NEW REWARD FUNCTION (TESTING) ********* ##        ((PPO_5))                                
    def set_previous_city_tile_count(self, amount=0):
        """
        Set the previous city tile count for each city for use in reward calculations.
        """
        self.previous_city_tile_count = amount

    def get_previous_city_tile_count(self):
        """
        Get the previous city tile count for a city.
        """
        return self.previous_city_tile_count


    def reward_for_city_survival(self, game):
        penalty_for_city_tile_loss = -2.0  # Severe penalty for losing city tiles

        # Calculate the current number of city tiles
        current_city_tile_count = sum(len(city.city_cells) for city in game.cities.values() if city.team == self.team)
        
        # Calculate the penalty for lost city tiles since last turn
        city_tiles_lost = max(0, self.previous_city_tile_count - current_city_tile_count)
        city_survival_reward = city_tiles_lost * penalty_for_city_tile_loss
        
        # Update the previous count for the next turn's calculation
        self.previous_city_tile_count = current_city_tile_count

        return city_survival_reward
    ## ********* NEW REWARD FUNCTION (TESTING) ********* ##

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return