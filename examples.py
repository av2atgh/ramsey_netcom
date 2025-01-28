import libs as rnc

save_to = "figs"
n = 50
seed = 0
params = {"model-": "ws", "K": 4, "p": 0.1}

rnc.draw_instance(n=n, params=params, seed=seed, path=save_to)

rnc.find_instances_with_communities(n=n, params=params, n_instances=3, path=save_to)

r_c = rnc.ramsey_community_number(params=params, epsilon=0.05, nr=10)
print(f"The Ramsey community number of {rnc.get_model_name(params)} is {r_c}")
