import numpy as np
import time
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

REMOVED = -1


def parse_dataset_to_mat(file_name, class_index):
    dataset = np.loadtxt(file_name, delimiter=",", skiprows=1)
    if file_name == "ad.data":
        dataset = dataset[~np.any(dataset == '?', axis=1)]
    y = np.transpose(dataset)[class_index, :]
    X = np.delete(dataset, class_index, axis=1)
    return X, y


class MLinUCB:

    def __init__(self, X, y, context_len, alpha, num_of_arms, m=1, missing_rewords_probability=0.25):
        self.X = X
        self.y = y
        self.T = X.shape[0]
        self.alpha = alpha
        self.c = context_len
        self.k = num_of_arms
        self.missing_rewords_probability = missing_rewords_probability
        self.A = np.stack([np.identity(self.c) for _ in range(self.k)], axis=0)
        self.b = np.stack([np.zeros(self.c) for _ in range(self.k)], axis=0)
        self.r = np.zeros(self.T)
        self.m = m
        self.N = 2
        self.remove_rewords()
        print("MLinUCB successfully initialized.")


    def remove_rewords(self, index_to_miss_from=20):
        ind = np.arange(self.T)
        removed_indices = np.random.choice(ind[index_to_miss_from:], int(np.floor(self.T * self.missing_rewords_probability)), replace=False)
        np.put(self.y, removed_indices, REMOVED)

    def compute_reward(self, t, a_t):
        y_t = self.y[t]
        if y_t == REMOVED:
            return self.compute_missing_reward(t)
        else:
            return y_t == (a_t + 1)


    def choose_arm(self, t, verbosity):
        x_t = self.X[t, :]
        A = self.A
        b = self.b
        p_t = np.zeros(self.k)

        for a in range(self.k):  # iterate over all arms
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_t) + self.alpha * np.sqrt(x_t.T.dot(A_a_inv).dot(x_t))

        max_p_t = np.nanmax(p_t)
        if max_p_t <= 0:
            print("User {} has max p_t={}, p_t={}".format(t, max_p_t, p_t))

        # randomly break ties, np.argmax return the first occurrence of maximum.
        # So I will get all occurrences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs) # a_t should be 1<= a_t <=k

        # observed reward = 1/0
        r_t = self.compute_reward(t, a_t)
        self.r[t] = r_t

        #if verbosity >= 2:
            #print("User {} choosing item {} with p_t={} reward {}".format(t, a_t, p_t[a_t], r_t))

        # update
        x_t_at = x_t
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        b[a_t] = b[a_t] + r_t * x_t_at.flatten()  # turn it back into an array because b[a_t] is an array

        return r_t

    def run_epoch(self, verbosity=2):
        """
        Call choose_arm() for each user in the dataset.
        :return: Average received reward.
        """
        rewards = []
        time_per_action = []
        start_time = time.time()

        for t in range(self.T):
            start_time_t = time.time()
            reward = self.choose_arm(t, verbosity)
            if not 0 < reward < 1:
                # not missing
                rewards.append(reward)
            time_t = time.time() - start_time_t
            time_per_action.append(time_t)
            #if verbosity >= 2:
                #print("Choosing arm for user {}/{} ended with reward {} in {}s".format(
                    #t, self.X[t], rewards[t], time_t))

        total_time = time.time() - start_time
        avg_reward = np.average(np.array(rewards))
        return avg_reward, total_time, rewards

    def run(self, num_epochs, verbosity=1):
        """
        Runs run_epoch() num_epoch times.
        :param verbosity:
        :param num_epochs: Number of epochs = iterating over all users.
        :return: List of average rewards per epoch.
        """

        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], total_time, rewards = self.run_epoch()
            print("Finished epoch {}/{} with avg reward {} in {}s".format(i, num_epochs, avg_rewards[i], total_time))

        return avg_rewards, rewards

    def compute_missing_reward(self, t):

        # compute clusters
        X_t = self.X[:t, :]
        x_t = self.X[t, :]
        r = self.r[:t]
        # visualizer = KElbowVisualizer(KMeans(), k=20, timings=False)
        # visualizer.fit(X_t)  # Fit the data to the visualizer
        # N = visualizer.elbow_value_
        # self.N = N if N is not None else self.N
        # TODO
        N = 2
        print("N is", N)
        kmeans = KMeans(n_clusters=N, random_state=0).fit(X_t)

        # calculate avg rewards and distance
        r_bar = np.zeros(N)
        d = np.zeros(N)
        clusters_prediction = kmeans.predict(X_t)
        for i in range(N):
            r_bar[i] = np.average(r[clusters_prediction == i])
            d[i] = np.linalg.norm(x_t - kmeans.cluster_centers_[i])

        # calculate reward
        m = self.m
        mask = d <= np.sort(d)[:m][-1]
        r_m = r_bar[mask]
        d_m = d[mask]

        return np.sum(r_m/d_m)/np.sum(np.ones(m)/d_m)