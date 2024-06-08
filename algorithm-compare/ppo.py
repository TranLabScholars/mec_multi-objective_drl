import torch

def run_ppo(env, num_episodes=1000, actor=None, critic=None):
    if actor is None or critic is None:
        print("Error: Actor and Critic models must be provided.")
        return [], []
    
    # Initialize lists to store delays and link utilizations
    delays = []
    link_utilisations = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            # Choose actions using the actor model
            action, _ = actor(torch.FloatTensor(obs))
            # print(action.detach().numpy())
            # Perform actions and observe next state and reward
            next_obs, _, done, _ = env.step(action.detach().numpy())
            # Update current observation
            obs = next_obs

        # Calculate average delay and link utilization for the episode
        avg_delay, avg_link_utilisation = env.estimate_performance()

        # Append to lists
        delays.append(avg_delay)
        link_utilisations.append(avg_link_utilisation)
        
        print(f"Episode {episode + 1}/{num_episodes}, Avg. Delay: {avg_delay}, Avg. Link Utilization: {avg_link_utilisation}")

    print("Average delay per task for each episode:", delays)
    print("Average link utilisation per task for each episode:", link_utilisations) 
    return delays, link_utilisations
