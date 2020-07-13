import torch
import torch.nn as nn


class WeightedLinearDynamics(nn.Module):

    def __init__(self, num_subsets: int, num_state_param: int, num_observation_param: int, num_action_param: int):
        super().__init__()

        self.dpn = nn.LSTM(num_state_param, num_subsets)

        self._K = num_subsets

        self._As = torch.randn(num_subsets, num_state_param, num_state_param,
                               requires_grad=True)
        self._Bs = torch.randn(num_subsets, num_state_param, num_action_param,
                               requires_grad=True)
        self._Cs = torch.randn(num_subsets, num_observation_param, num_state_param,
                               requires_grad=True)

    @property
    def num_subsets(self) -> int:
        return self._K

    def forward(self, state: torch.Tensor) -> tuple:

        # Reshape tensor to (series_length, batch_size, num_state_param)
        if state.dim() == 1:
            state = state.view(1, 1, -1)
        elif state.dim() == 2:
            state = state.view(1, *state.shape)
        elif state.dim() == 3:
            pass
        else:
            raise Exception('Expected state tensor with either 1, 2 or 3 dimensions!')

        weighting = self.dpn(state)[0]
        weighting = weighting.view(*weighting.shape, 1, 1)

        batch_As = torch.cat((self._As.unsqueeze(0),) * state.size(1), dim=0)
        series_As = torch.cat((batch_As.unsqueeze(0),) * state.size(0), dim=0)

        batch_Bs = torch.cat((self._Bs.unsqueeze(0),) * state.size(1), dim=0)
        series_Bs = torch.cat((batch_Bs.unsqueeze(0),) * state.size(0), dim=0)

        batch_Cs = torch.cat((self._Cs.unsqueeze(0),) * state.size(1), dim=0)
        series_Cs = torch.cat((batch_Cs.unsqueeze(0),) * state.size(0), dim=0)

        A = torch.sum(weighting * series_As, dim=2)
        B = torch.sum(weighting * series_Bs, dim=2)
        C = torch.sum(weighting * series_Cs, dim=2)

        return A, B, C


if __name__ == '__main__':

    _num_state_params = 15
    _num_observation_params = 10
    _num_action_params = 4
    _bs = 10
    _ts = 6

    _state_vector = torch.randn(_num_state_params)
    _state_batch = torch.randn(_bs, _num_state_params)
    _state_series = torch.randn(_ts, _bs, _num_state_params)

    _wld = WeightedLinearDynamics(7, _num_state_params, _num_observation_params, _num_action_params)

    _wld(_state_vector)
    _wld(_state_batch)
    _wld(_state_series)









