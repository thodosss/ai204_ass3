import numpy as np
import random
from collections import defaultdict

class RecsysEnv:
    def __init__(self, user_sequences, ratings_df=None, seq_len=5):
        """
        user_sequences: dict of user interaction sequences, e.g. {user_id: [item_id1, item_id2, ...]}
        ratings_df: DataFrame containing user ratings (optional)
        seq_len: number of past items to use as history
        """
        self.user_sequences = user_sequences
        self.ratings_df = ratings_df
        self.seq_len = seq_len
        
        # Create rating lookup dictionary for faster access
        self.ratings_dict = defaultdict(dict)
        if ratings_df is not None:
            for _, row in ratings_df.iterrows():
                self.ratings_dict[row['user_id']][row['item_id']] = row['rating']
        
        # Calculate user statistics
        self.user_stats = self._calculate_user_statistics()
        
        self.reset()

    def _calculate_user_statistics(self):
        """Calculate statistics for each user"""
        stats = {}
        for user_id, seq in self.user_sequences.items():
            # Get user's ratings if available
            user_ratings = self.ratings_dict[user_id] if user_id in self.ratings_dict else {}
            
            stats[user_id] = {
                'avg_rating': np.mean(list(user_ratings.values())) if user_ratings else 0,
                'num_interactions': len(seq),
                'unique_items': len(set(seq)),
                'diversity': len(set(seq)) / len(seq) if seq else 0,  # Ratio of unique items to total items
                'recency': 0  # Will be updated during episode
            }
        return stats

    def _get_state_features(self):
        """Get enhanced state features"""
        # Basic sequence features
        seq_features = np.zeros(self.seq_len)
        if len(self.history) > 0:
            seq_features[-len(self.history):] = self.history[-self.seq_len:]
        
        # User statistics
        user_stats = self.user_stats[self.user_id]
        user_features = np.array([
            user_stats['avg_rating'],
            user_stats['num_interactions'] / 1000,  # Normalize
            user_stats['diversity'],
            user_stats['recency']
        ])
        
        # Temporal features
        temporal_features = np.array([
            self.position / len(self.user_seq),  # Progress through sequence
            len(self.history) / self.seq_len,    # History fullness
        ])
        
        # Combine all features
        return np.concatenate([
            seq_features,
            user_features,
            temporal_features
        ])

    def reset(self):
        self.user_id = random.choice(list(self.user_sequences.keys()))
        self.user_seq = self.user_sequences[self.user_id]
        self.position = 0
        self.done = False
        self.history = []
        self.last_recommendations = []  # Track last few recommendations for diversity
        
        # Update recency for this user
        self.user_stats[self.user_id]['recency'] = 0
        
        return self._get_state_features()

    def _get_rating_reward(self, item_id):
        """Get reward based on user's rating for the item"""
        if self.ratings_dict and self.user_id in self.ratings_dict and item_id in self.ratings_dict[self.user_id]:
            # Normalize rating to [0, 1] range (assuming ratings are 1-5)
            return (self.ratings_dict[self.user_id][item_id] - 1) / 4
        return 0.5  # Default reward if no rating available

    def _get_novelty_reward(self, item_id):
        """Penalize recommending items the user has already seen"""
        if item_id in self.history:
            return -0.5  # Penalty for recommending seen items
        return 0.0

    def _get_diversity_reward(self, item_id):
        """Penalize recommending similar items in sequence"""
        if len(self.last_recommendations) > 0 and item_id in self.last_recommendations[-3:]:
            return -0.3  # Penalty for recommending similar items recently
        return 0.0

    def _get_time_reward(self):
        """Reward based on recency of interactions"""
        if len(self.history) > 0:
            # Reward decreases as we get further in the sequence
            return 1.0 - (self.position / len(self.user_seq))
        return 1.0

    def step(self, action):
        if self.done:
            raise Exception("Episode done, call reset() before step()")

        # Check if current position is valid
        if self.position >= len(self.user_seq):
            self.done = True
            return None, 0, True, self._get_state_features()

        actual_next_item = self.user_seq[self.position]
        
        # Calculate different reward components
        rating_reward = self._get_rating_reward(action)
        novelty_reward = self._get_novelty_reward(action)
        diversity_reward = self._get_diversity_reward(action)
        time_reward = self._get_time_reward()
        
        # Combine rewards with weights
        reward = (
            0.4 * rating_reward +  # Rating-based reward
            0.3 * (1 if action == actual_next_item else 0) +  # Accuracy reward
            0.2 * novelty_reward +  # Novelty reward
            0.1 * diversity_reward  # Diversity reward
        ) * time_reward  # Time-based scaling

        # Update state
        self.history.append(actual_next_item)
        self.last_recommendations.append(action)
        if len(self.last_recommendations) > 5:  # Keep only last 5 recommendations
            self.last_recommendations.pop(0)
        
        # Update recency
        self.user_stats[self.user_id]['recency'] = len(self.history) / len(self.user_seq)
        
        self.position += 1

        if self.position >= len(self.user_seq):
            self.done = True

        return actual_next_item, reward, self.done, self._get_state_features()

''' 
import numpy as np
import random
from collections import defaultdict

class RecsysEnv:
    def __init__(self, user_sequences, ratings_df=None, seq_len=5):
        """
        user_sequences: dict of user interaction sequences, e.g. {user_id: [item_id1, item_id2, ...]}
        ratings_df: DataFrame containing user ratings (optional)
        seq_len: number of past items to use as history
        """
        self.user_sequences = user_sequences
        self.ratings_df = ratings_df
        self.seq_len = seq_len
        
        # Create rating lookup dictionary for faster access
        self.ratings_dict = defaultdict(dict)
        if ratings_df is not None:
            for _, row in ratings_df.iterrows():
                self.ratings_dict[row['user_id']][row['item_id']] = row['rating']
        
        # Calculate user statistics
        self.user_stats = self._calculate_user_statistics()
        
        self.reset()

    def _calculate_user_statistics(self):
        """Calculate statistics for each user"""
        stats = {}
        for user_id, seq in self.user_sequences.items():
            # Get user's ratings if available
            user_ratings = self.ratings_dict[user_id] if user_id in self.ratings_dict else {}
            
            stats[user_id] = {
                'avg_rating': np.mean(list(user_ratings.values())) if user_ratings else 0,
                'num_interactions': len(seq),
                'unique_items': len(set(seq)),
                'diversity': len(set(seq)) / len(seq) if seq else 0,  # Ratio of unique items to total items
                'recency': 0  # Will be updated during episode
            }
        return stats

    def _get_state_features(self):
        """Get enhanced state features"""
        # Basic sequence features
        seq_features = np.zeros(self.seq_len)
        if len(self.history) > 0:
            seq_features[-len(self.history):] = self.history[-self.seq_len:]
        
        # User statistics
        user_stats = self.user_stats[self.user_id]
        user_features = np.array([
            user_stats['avg_rating'],
            user_stats['num_interactions'] / 1000,  # Normalize
            user_stats['diversity'],
            user_stats['recency']
        ])
        
        # Temporal features
        temporal_features = np.array([
            self.position / len(self.user_seq),  # Progress through sequence
            len(self.history) / self.seq_len,    # History fullness
        ])
        
        # Combine all features
        return np.concatenate([
            seq_features,
            user_features,
            temporal_features
        ])

    def reset(self):
        self.user_id = random.choice(list(self.user_sequences.keys()))
        self.user_seq = self.user_sequences[self.user_id]
        self.position = 0
        self.done = False
        self.history = []
        self.last_recommendations = []  # Track last few recommendations for diversity
        
        # Update recency for this user
        self.user_stats[self.user_id]['recency'] = 0
        
        return self._get_state_features()

    def _get_rating_reward(self, item_id):
        """Get reward based on user's rating for the item"""
        if self.ratings_dict and self.user_id in self.ratings_dict and item_id in self.ratings_dict[self.user_id]:
            # Normalize rating to [0, 1] range (assuming ratings are 1-5)
            return (self.ratings_dict[self.user_id][item_id] - 1) / 4
        return 0.5  # Default reward if no rating available

    def _get_novelty_reward(self, item_id):
        """Penalize recommending items the user has already seen"""
        if item_id in self.history:
            return -0.5  # Penalty for recommending seen items
        return 0.0

    def _get_diversity_reward(self, item_id):
        """Penalize recommending similar items in sequence"""
        if len(self.last_recommendations) > 0 and item_id in self.last_recommendations[-3:]:
            return -0.3  # Penalty for recommending similar items recently
        return 0.0

    def _get_time_reward(self):
        """Reward based on recency of interactions"""
        if len(self.history) > 0:
            # Reward decreases as we get further in the sequence
            return 1.0 - (self.position / len(self.user_seq))
        return 1.0

    def step(self, action):
        if self.done:
            raise Exception("Episode done, call reset() before step()")

        # Check if current position is valid
        if self.position >= len(self.user_seq):
            self.done = True
            return None, 0, True, self._get_state_features()

        actual_next_item = self.user_seq[self.position]
        
        # Calculate different reward components
        rating_reward = self._get_rating_reward(action)
        novelty_reward = self._get_novelty_reward(action)
        diversity_reward = self._get_diversity_reward(action)
        time_reward = self._get_time_reward()
        
        # Combine rewards with weights
        reward = (
            0.4 * rating_reward +  # Rating-based reward
            0.3 * (1 if action == actual_next_item else 0) +  # Accuracy reward
            0.2 * novelty_reward +  # Novelty reward
            0.1 * diversity_reward  # Diversity reward
        ) * time_reward  # Time-based scaling

        # Update state
        self.history.append(actual_next_item)
        self.last_recommendations.append(action)
        if len(self.last_recommendations) > 5:  # Keep only last 5 recommendations
            self.last_recommendations.pop(0)
        
        # Update recency
        self.user_stats[self.user_id]['recency'] = len(self.history) / len(self.user_seq)
        
        self.position += 1

        if self.position >= len(self.user_seq):
            self.done = True

        return actual_next_item, reward, self.done, self._get_state_features()


'''