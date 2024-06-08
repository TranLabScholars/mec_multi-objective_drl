import numpy as np

# Define the softmax function
def softmax(x, tau):
    exp_x = np.exp(x / tau)
    return exp_x / np.sum(exp_x)

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

def run_softmax(env, observation_space, num_episodes, alpha=0.1, gamma=0.99, tau=0.1):
    """
    Run the softmax policy gradient algorithm to learn a policy for the given environment.

    Args:
    - env: The environment to run the algorithm on.
    - observation_space: The observation space of the servers.
    - num_episodes: The number of episodes to run the algorithm for.
    - alpha: The learning rate (default is 0.1).
    - gamma: The discount factor (default is 0.99).
    - tau: The temperature parameter for the softmax function (default is 0.1).

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
            # Preprocess the current state
            processed_state = preprocess_state(state)
            
            # Calculate the final probabilities using the softmax function
            action_probs = softmax(processed_state, tau)
            
            # Sample action from the softmax distribution
            action = action_probs
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
