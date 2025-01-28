import libs as rnc

# folder where the figures will be saved

save_to = "figs"

# a choice of network size

n = 50

# draw instances, specified by the parameter seed = 0, 1, ...

seed = 0

rnc.draw_instance(n, {"model-": "ws", "K": 2, "p": 0.0}, seed, save_to)
rnc.draw_instance(n, {"model-": "ws", "K": 4, "p": 0.0}, seed, save_to)

# draw instances with communities, using buildin method

#rnc.find_instances_with_communities(n, {"model-": "ls", "d": 1}, 10, save_to)
#rnc.find_instances_with_communities(n, {"model-": "ds", "q": 0.3}, 10, save_to)
#rnc.find_instances_with_communities(n, {"model-": "ws", "K": 4, "p": 0.1}, 10, save_to)

# estimate ramsey community number

params = {"model-": "ls", "d": 1}
r_c = rnc.ramsey_community_number(params=params, epsilon=0.05, nr=100)
print(f"The Ramsey community number of {rnc.get_model_name(params)} is {r_c}")
