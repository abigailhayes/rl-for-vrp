import vrplib
from operator import itemgetter

data = vrplib.read_instance('instances/A/A-n32-k5.vrp')

def cw_savings(data):
    # Initialise to a single node in each route
    routes = [[i] for i in range(1,data['dimension'])]

    # Calculate savings
    def saving(i,j):
        # Calculate saving for a specific node pair
        return data['edge_weight'][i][0]+data['edge_weight'][0][j]-data['edge_weight'][i][j]

    savings = []
    for i in range(2,data['dimension']):
        for j in range(1, i):
            savings.append((i,j,saving(i,j)))
    savings.sort(key=itemgetter(2), reverse=True)

    # Run through list of savings and merge routes where appropriate
    def get_route(i):
        # Get the current route a node is in
        return [r for r in routes if i in r][0]

    def pos_check(i):
        # Check where in a route a node appears
        route = get_route(i)
        if route[0]==i:
            return 0
        elif route[-1]==i:
            return 1
        else:
            return 2

    def merge_routes(i,j):
        # Merge routes based on positioning of nodes
        if pos_check(i)<pos_check(j):
            return get_route(j)+get_route(i)
        elif pos_check(i)<pos_check(j):
            return get_route(i) + get_route(j)
        else:
            return get_route(i) + list(reversed(get_route(j)))

    def cap_check(new_route):
        return sum([data['demand'][i] for i in new_route])

    for i,j,c in savings:
        if pos_check(i)==2 or pos_check(j)==2:
            continue
        new_route = merge_routes(i,j)
        if cap_check(new_route)>data['capacity']:
            continue
        routes = [r for r in routes if i not in r and j not in r]
        routes.append(new_route)

    return routes




