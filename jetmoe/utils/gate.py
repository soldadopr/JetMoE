import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGating(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts, "top_k cannot be greater than num_experts"
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return f"top_k={self.top_k}, num_experts={self.num_experts}"

    def compute_aux_loss(self, probs, logits, gates):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            probs (torch.Tensor): Probability values for each expert.
            logits (torch.Tensor): Logits for each expert.
            gates (torch.Tensor): Gates tensor.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        count = logits.size(0)
        probs_sum = probs.sum(0)
        freq = (gates > 0).float().sum(0)
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1)) ** 2).sum()

        switch_loss = self.num_experts * (F.normalize(probs_sum, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        z_loss = lsesq / count
        loss = switch_loss + 0.1 * z_loss

        return loss

    def forward(self, x):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
        """
        logits = self.layer(x).float()
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(x)

        if self.training:
            probs = torch.softmax(logits, dim=1)
            zeros = torch.zeros_like(probs)
            zeros = zeros.to(top_k_gates.dtype)  # Convert zeros to match top_k_gates dtype
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
            self.loss = self.compute_aux_loss(probs, logits, gates)
        else:
            self.loss = torch.tensor(0.0, device=x.device)

        return top_k_indices, top_k_gates
