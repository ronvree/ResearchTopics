

class Episode:

    # TODO -- append reward, append observation etc functions that take into account the correct ordering?

    """

        Convenience class for storing episodes

        Episodes are stored as sequences of:

        o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T

        where
            o denotes an observation
            a denotes an action
            r denotes a reward

        o_t denotes the observation at time step t

        All entries are stored as tensors (without batch dimension)


    """

    def __init__(self):
        self._episode = []

    def __len__(self):
        return len(self._episode)

    def __getitem__(self, index: int):
        return self._episode[index]

    @staticmethod
    def from_tuple(episode: tuple) -> "Episode":
        """
        Create a new Episode from a data sequence of
        o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        :param episode: data sequence tuple
        :return: an Episode object containing the data
        """
        e = Episode()
        e.append_all(episode)
        return e

    def as_tuple(self) -> tuple:
        """
        Get the episode data in a tuple
        :return: a tuple containing a data sequence of
                    o_0, a_0, r_1, o_1, a_1, ... , o_{T - 1}, a_{T - 1}, r_T, o_T
        """
        return tuple(self._episode)

    def append(self, tensor):
        """
        Append an entry to this episode
        :param tensor: the entry to be added
        """
        self._episode.append(tensor)

    def append_all(self, *tensors):
        """
        Append multiple entries to this episode
        :param tensors: the entries to be added
        """
        self._episode.extend([*tensors])

    @property
    def observations(self) -> tuple:
        pass  # TODO

    @property
    def actions(self) -> tuple:
        pass  # TODO

    @property
    def rewards(self) -> tuple:
        pass  # TODO

