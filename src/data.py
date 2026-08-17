import sqlite3
import pandas as pd

conn = sqlite3.connect("./data/kilter_splits.sqlite")

placements = dict(pd.read_sql("SELECT id, hole_id FROM placements", conn).values)
roles = dict(pd.read_sql("SELECT id, name FROM placement_roles", conn).values)
hold_data = pd.read_sql("SELECT id, x, y FROM holes", conn).values
holds = {row[0]: (row[1], row[2]) for row in hold_data}

def get_holds(route):

    """
    Parses a route frame into a list of holds with their board position and role.

    Args:
        route: row from kilter_train/val/test, must contain 'frames'.

    Returns:
        List of (x, y, role) tuples, one per hold in the frame.
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

def get_avg_nearest_hold_dist(route_holds):

    """
    Calculates the average distance between the nearest holds of the given route.

    Args:
        route_holds: list of (x, y) tuples.

    Returns:
        Average distance from each hold to its closest other hold.
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

def get_max_nearest_hold_dist(route_holds):

    """
    Calculates the max distance between the nearest holds of the given route.

    Args:
        route_holds: list of (x, y) tuples.

    Returns:
        Max distance between nearest holds.
    """

    max_distance = 0

    for h in route_holds:
        closest_dist = None
        for j in route_holds:
            if j is h:
                continue
            dist = ((h[0] - j[0]) ** 2 + (h[1] - j[1]) ** 2) ** 0.5
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
        if closest_dist > max_distance:
            max_distance = closest_dist

    return max_distance

def get_route_width(route_holds):

    """
    Calculates the width of the route

    Args:
        route_holds: list of (x, y) tuples.

    Returns:
        Width of the route
    """

    min_x = None
    max_x = None

    for h in route_holds:
        if max_x is None or h[0] > max_x:
            max_x = h[0]
        if min_x is None or h[0] < min_x:
            min_x = h[0]

    return max_x - min_x

def get_route_height(route_holds):

    """
    Calculates the height of the route

    Args:
        route_holds: list of (x, y) tuples.

    Returns:
        Height of the route
    """

    min_y = None
    max_y = None

    for h in route_holds:
        if max_y is None or h[1] > max_y:
            max_y = h[1]
        if min_y is None or h[1] < min_y:
            min_y = h[1]

    return max_y - min_y

def get_features(route):

    """
    Converts the given route to a feature dict that can be used as one row of the training table.

    Args:
        route: row from kilter_train/val/test, must contain 'frames'.

    Returns:
        Feature dictionary containing the angle, number of holds, hold
        counts by type, and average nearest-hold distance.
    """

    route_angle = int(route['angle'])
    route_holds = get_holds(route)

    role_list = [r[2] for r in route_holds]
    num_start = role_list.count('start')
    num_middle = role_list.count('middle')
    num_foot = role_list.count('foot')
    num_finish = role_list.count('finish')

    no_foot_holds = [(x, y) for x, y, role in route_holds if role != 'foot']

    avg_nearest_hold_dist = get_avg_nearest_hold_dist(no_foot_holds)
    max_nearest_hold_dist = get_max_nearest_hold_dist(no_foot_holds)

    route_width = get_route_width(route_holds)
    route_height = get_route_height(route_holds)

    num_holds = len(route_holds)
    foot_ratio = num_foot / num_holds
    middle_ratio = num_middle / num_holds

    return {
        "angle": route_angle,
        "num_holds": num_holds,
        "num_start": num_start,
        "num_middle": num_middle,
        "num_foot": num_foot,
        "num_finish": num_finish,
        "avg_nearest_hold_dist": avg_nearest_hold_dist,
        "max_nearest_hold_dist": max_nearest_hold_dist,
        "route_width": route_width,
        "route_height": route_height,
        "foot_ratio": foot_ratio,
        "middle_ratio": middle_ratio
    }


def process_data(data_type):

    """
    Create a csv with a list of the features of each route of the chosen data type.

    Args:
        data_type: "test", "train" or "val".

    Raises:
        ValueError: if data_type is not one of the three valid options.
    """

    if data_type not in ("test", "train", "val"):
        raise ValueError("Invalid parameter. Valid: test, train, val")

    routes = pd.read_sql(f"SELECT * FROM kilter_{data_type}", conn)

    res = []
    errors = []

    for _, row in routes.iterrows():
        try:
            features = get_features(row)
            features["grade"] = row["difficulty_numeric"]
            res.append(features)
        except Exception as e:
            errors.append((row['uuid'], str(e)))

    print(f"processed rows: {len(res)}, errors: {len(errors)}")

    data = pd.DataFrame(res)
    data.to_csv(f"data/{data_type}_features.csv", index=False)


if __name__ == "__main__":

    process_data("train")
    process_data("val")
    process_data("test")