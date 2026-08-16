import sqlite3
import pandas as pd

conn = sqlite3.connect("./data/kilter_splits.sqlite")

placements = dict(pd.read_sql("SELECT id, hole_id FROM placements", conn).values)
roles = dict(pd.read_sql("SELECT id, name FROM placement_roles", conn).values)
hold_data = pd.read_sql("SELECT id, x, y FROM holes", conn).values
holds = {row[0]: (row[1], row[2]) for row in hold_data}

def getHolds(route):

    """
    Parses a route frame into a list of holds with their board position and role.

    Args: Route, row from kilter_train/val/test, must contain 'frames'.
    Returns: List of (x, y, role) tuples, one per hold in the frame.
    """

    # Format: p<placement_id>r<role_id>...
    p = route['frames'].split('p')[1:]
    pos_role = [parte.split('r') for parte in p]

    res = []
    for placement_id_str, role_id_str in pos_role:
        placement_id = int(placement_id_str)
        role_id = int(role_id_str)
        hold_id = placements[placement_id]
        x = int(holds[hold_id][0])
        y = int(holds[hold_id][1])
        role = roles[role_id]

        res.append((x, y, role))

    return res

def getAvgDist(route_holds):

    """
    Calculates the average distance between the holds of the given route

    Args: route holds, list of (x, y) tuples
    Returns: Average distance from each hold to its closest other hold
    """

    distances = []

    for h in route_holds:
        closest_dist = None
        for j in route_holds:
            if j is h:
                continue
            dist = ((h[0] - j[0]) ** 2 + (h[1] - j[1]) ** 2) ** 0.5
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
        distances.append(closest_dist)

    return sum(distances) / len(distances)




def getFeatures(route):

    """
    Converts the given route to a feature dict that can be used as one row of the training table.

    Args: Route, row from kilter_train/val/test, must contain 'frames'.
    Returns: Feature dictionary: Angle, number of holds, hold count by type, and the avg distance 
        between holds.

    """

    route_angle = int(route['angle'])
    route_holds = getHolds(route)

    role_list = [r[2] for r in route_holds]
    num_start = role_list.count('start')
    num_middle = role_list.count('middle')
    num_foot = role_list.count('foot')
    num_finish = role_list.count('finish')

    avg_dist = getAvgDist([(x, y) for x, y, role in route_holds if role != 'foot'])

    return {
        "angle": route_angle,
        "num_holds": len(route_holds),
        "num_start": num_start,
        "num_middle": num_middle,
        "num_foot": num_foot,
        "num_finish": num_finish,
        "avg_dist": avg_dist,
    }


def process_data(data_type):

    """
    Create a csv with a list of the features of each route of the chosen data type.

    Args: Data type: test, train or val
    Raises: ValueError if data_type is not one of the three valid options.
    """

    if data_type not in ("test", "train", "val"):
        raise ValueError("Invalid parameter. Valid: test, train, val")

    routes = pd.read_sql(f"SELECT * FROM kilter_{data_type}", conn)

    res = []
    errors = []

    for _, row in routes.iterrows():
        try:
            features = getFeatures(row)
            features["grade"] = row["difficulty_numeric"]
            res.append(features)
        except Exception as e:
            errors.append((row['uuid'], str(e)))

    print(f"filas procesadas: {len(res)}, errores: {len(errors)}")

    data = pd.DataFrame(res)
    data.to_csv(f"data/{data_type}_features.csv", index=False)


process_data("train")
process_data("val")
process_data("test")