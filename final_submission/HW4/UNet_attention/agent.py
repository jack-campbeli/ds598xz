import os
import numpy as np
import torch
from lux.game import Game

path = os.path.dirname(os.path.abspath(__file__))

# load the model
# change the path if you need
model = torch.jit.load(f'{path}/model_all_top_agents_0.84.pth')
model.eval()

# function to get input state
def make_input(obs, game_state):
    width, height = game_state.map.width, game_state.map.height
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    cities_opp = {}

    b = np.zeros((14, 32, 32), dtype=np.float32)
    b_global = np.zeros((15, 4, 4), dtype=np.float32)

    global_unit = 0
    global_rp = 0
    global_city = 0
    global_citytile = 0

    global_unit_opp = 0
    global_rp_opp = 0
    global_city_opp = 0
    global_citytile_opp = 0

    global_wood = 0
    global_coal = 0
    global_uranium = 0

    # in some of our models we used global resources features
    # global_wood_amount = 0
    # global_coal_amount = 0
    # global_uranium_amount = 0

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            team = int(strs[2])
            cooldown = float(strs[6])
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if team == obs.player:
                b[0, x, y] = 1  # b0 friend unit
                global_unit += 1
                b[1, x, y] = cooldown / 6   # b1 friend cooldown
                b[2, x, y] = (wood + coal + uranium) / 100  # b2 friend cargo
            else:
                b[3, x, y] = 1  # b3 oppo unit
                global_unit_opp += 1
                b[4, x, y] = cooldown / 6   # b4 oppo cooldown
                b[5, x, y] = (wood + coal + uranium) / 100  # b5 oppo cargo

        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            if team == obs.player:
                global_citytile += 1
                b[6, x, y] = 1  # b6 friend city
                b[7, x, y] = cities[city_id]   # b7 friend city nights to survive
            else:
                global_citytile_opp += 1
                b[8, x, y] = 1  # b8 oppo city
                b[9, x, y] = cities_opp[city_id]  # b9 oppo city nights to survive
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 10, 'coal': 11, 'uranium': 12}[r_type], x, y] = amt / 800
            if r_type == 'wood':
                global_wood += 1
            elif r_type == "coal":
                global_coal += 1
            else:
                global_uranium += 1
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            if team == obs.player:
                global_rp = min(rp, 200) / 200
            else:
                global_rp_opp = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            team = int(strs[1])
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            if team == obs.player:
                global_city += 1
                cities[city_id] = min(fuel / lightupkeep, 20) / 20
            else:
                global_city_opp += 1
                cities_opp[city_id] = min(fuel / lightupkeep, 20) / 20
    # Map Size
    b[13, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
    # global features (normalized)
    b_global[0, :, :] = global_unit / width / height
    b_global[1, :, :] = global_rp
    b_global[2, :, :] = global_city / width / height
    b_global[3, :, :] = global_citytile / width / height
    b_global[4, :, :] = np.array(list(cities.values())).mean() if cities else 0
    b_global[5, :, :] = global_unit_opp / width / height
    b_global[6, :, :] = global_rp_opp
    b_global[7, :, :] = global_city_opp / width / height
    b_global[8, :, :] = global_citytile_opp / width / height
    b_global[9, :, :] = np.array(list(cities_opp.values())).mean() if cities_opp else 0
    b_global[10, :, :] = global_wood / width / height
    b_global[11, :, :] = global_coal / width / height
    b_global[12, :, :] = global_uranium / width / height
    # b_global[13, :, :] = (global_wood_amount / global_wood if global_wood_amount != 0 else 0)
    # b_global[14, :, :] = (global_coal_amount / global_coal if global_coal_amount != 0 else 0)
    # b_global[15, :, :] = (global_uranium_amount / global_uranium if global_uranium_amount != 0 else 0)
    b_global[13, :, :] = obs['step'] % 40 / 40  # Day/Night Cycle
    b_global[14, :, :] = obs['step'] / 360  # Turns

    return b, b_global


game_state = None
def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    return game_state

def get_shift(game_state):
    width, height = game_state.map.width, game_state.map.height
    shift = (32 - width) // 2
    return shift

def in_city(pos):    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]

def get_action(policy, unit, dest, shift):
    # policy is a predicted action map, we get the worker's action according to its position
    p = policy[:, unit.pos.x + shift, unit.pos.y + shift]

    for label in np.argsort(p)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos
        if pos not in dest or in_city(pos):
            return call_func(unit, *act), pos

    return unit.move('c'), unit.pos

def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)    
    shift = get_shift(game_state)
    player = game_state.players[observation.player]
    actions = []
    
    # City Actions
    unit_count = len(player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                if unit_count < player.city_tile_count: 
                    actions.append(city_tile.build_worker())
                    unit_count += 1
                elif not player.researched_uranium():
                    actions.append(city_tile.research())
                    player.research_points += 1
    
    # Worker Actions
    state_1, state_2 = make_input(observation, game_state)
    dest = []
    with torch.no_grad():
        p = model(torch.from_numpy(state_1).unsqueeze(0), torch.from_numpy(state_2).unsqueeze(0))
        policy = p.squeeze(0).numpy()
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            action, pos = get_action(policy, unit, dest, shift)
            actions.append(action)
            dest.append(pos)
            
    return actions
