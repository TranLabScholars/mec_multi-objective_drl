import numpy as np

def preprocess_state(state):
    """
    Preprocesses the current state of each server.

    Parameters:
    - state: The current state of each server.

    Returns:
    - processed_state: The preprocessed state of each server.
    """
    # Calculate the mean of each column (parameter) across all servers
    mean_state = np.mean(state, axis=0)
    return mean_state

def run_egreedy(env, observation_space, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Runs the epsilon-greedy algorithm for reinforcement learning with a continuous action space.

    Parameters:
    - env: The environment.
    - observation_space: The observation space of the servers.
    - num_episodes: Number of episodes to run the algorithm.
    - alpha: Learning rate.
    - gamma: Discount factor for future rewards.
    - epsilon: The probability of choosing a random action (exploration rate).

    Returns:
    - avg_delays: A list containing the average delay per task for each episode.
    - avg_link_utilisations: A list containing the average link utilisation per task for each episode.
    """
    num_states = observation_space.shape[0]  # Number of states
    num_actions = observation_space.shape[1]  # Number of actions
    
    Q = np.zeros((num_states, num_actions))  # Initialize Q-table 

    avg_delays = []  # List to store average delay per task for each episode
    avg_link_utilisations = []  # List to store average link utilisation per task for each episode
    
    for _ in range(num_episodes):
        state = env.reset()  # Reset the environment to initial state
        done = False  # Initialize flag indicating whether the episode has ended
        
        while not done:
            if np.random.rand() < epsilon:  # With probability epsilon, choose a random action
                # Sample from a normal distribution centered around 0 with a small standard deviation
                action = np.random.binomial(n=1, p=0.5, size=num_actions)
            else:
                # Choose action greedily based on current policy
                # Preprocess the current state
                processed_state = preprocess_state(state)
                # Sample action from a Bernoulli distribution with the mean of Q-values as the probability parameter
                mean_action = Q[processed_state.astype(int)]
                action = np.random.binomial(n=1, p=mean_action)
            print(action)
            next_state, reward, done, _ = env.step(action)  # Take a step in the environment

            if state.shape[0] in range(num_states) and action in range(num_actions):
                # Update Q-value using the Q-learning update rule
                Q[processed_state.astype(int), action] += alpha * (reward + gamma * np.max(Q[next_state.astype(int)]) - Q[processed_state.astype(int), action])
            
            state = next_state # Move to the next state
        
        # Calculate average delay and link utilisation for this episode
        avg_delay, avg_link_utilisation = env.estimate_performance()
        
        avg_delays.append(avg_delay)  # Append the average delay per task for this episode to the list
        avg_link_utilisations.append(avg_link_utilisation)  # Append the average link utilisation per task for this episode to the list
        
    print("Average delay per task for each episode:", avg_delays)
    print("Average link utilisation per task for each episode:", avg_link_utilisations)
    return avg_delays, avg_link_utilisations
