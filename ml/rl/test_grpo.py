"""Numerical contracts for GRPO; run with python -m unittest discover -s rl."""

import math
import unittest

import torch

from rl.grpo import grpo_advantage, grpo_loss


# Miles's sample-std normalization of rewards [1, 0], including epsilon=1e-6.
BINARY_ADVANTAGE = 0.707105756


class GRPOTests(unittest.TestCase):
    def test_advantages_are_relative_within_each_prompt(self):
        cases = [
            (
                [[1.0, 0.0, 1.0, 0.0], [12.0, 10.0, 12.0, 10.0], [5.0] * 4],
                [
                    [0.866023904, -0.866023904] * 2,
                    [0.866024654, -0.866024654] * 2,
                    [0.0] * 4,
                ],
            ),
            ([[1.0, 0.0]], [[BINARY_ADVANTAGE, -BINARY_ADVANTAGE]]),
            ([[0.1], [1.0]], [[0.0], [0.0]]),
            ([[0.0, 1e-7]], [[-0.046697956, 0.046697956]]),
        ]
        for values, expected in cases:
            with self.subTest(rewards=values):
                rewards = torch.tensor(values, requires_grad=True)
                advantages = grpo_advantage(rewards)
                torch.testing.assert_close(advantages, torch.tensor(expected))
                self.assertFalse(advantages.requires_grad)

    def test_clipping_is_directional(self):
        # Per row: stop already-good moves; allow recovery from bad moves;
        # and retain ordinary policy gradients inside the clipping interval.
        ratios = torch.tensor([[1.5, 0.5], [0.5, 1.5], [1.1, 0.9]])
        current = (ratios * 0.25).log().unsqueeze(-1).requires_grad_()
        old = torch.full_like(current, math.log(0.25), requires_grad=True)
        rewards = torch.tensor([[1.0, 0.0]] * 3, requires_grad=True)
        loss = grpo_loss(
            current, old, rewards, torch.ones_like(current, dtype=torch.bool)
        )

        # Surrogates, before advantage scaling: [1.2, -0.8, 0.5, -1.5, 1.1, -0.9].
        torch.testing.assert_close(loss, torch.tensor(BINARY_ADVANTAGE / 15.0))
        loss.backward()
        expected_grad = torch.tensor([[0.0, 0.0], [-0.5, 1.5], [-1.1, 0.9]]) * (
            BINARY_ADVANTAGE / 6
        )
        torch.testing.assert_close(current.grad.squeeze(-1), expected_grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(rewards.grad)

    def test_extreme_ratios_and_zero_advantages_have_finite_gradients(self):
        cases = [
            ([-1.0, -1.0], [-101.0, -1.0], [1.0, 0.0], -0.1, [0.0, 0.5]),
            ([-1.0, -1.0], [-101.0, -101.0], [0.0, 0.0], 0.0, [0.0, 0.0]),
            # An unfavorable increase is capped before exp can overflow.
            (
                [-1.0, -1.0],
                [-101.0, -1.0],
                [0.0, 1.0],
                (math.exp(20) - 1) / 2,
                [0.0, -0.5],
            ),
            # Below the lower safety bound, the otherwise active gradient is zero.
            (
                [-31.0, -1.0],
                [-1.0, -1.0],
                [1.0, 0.0],
                (1 - math.exp(-20)) / 2,
                [0.0, 0.5],
            ),
        ]
        for current_values, old_values, rewards, expected_loss, expected_grad in cases:
            with self.subTest(current=current_values, old=old_values, rewards=rewards):
                current = torch.tensor([current_values]).unsqueeze(-1).requires_grad_()
                old = torch.tensor([old_values]).unsqueeze(-1)
                loss = grpo_loss(
                    current,
                    old,
                    torch.tensor([rewards]),
                    torch.ones_like(current, dtype=torch.bool),
                )
                torch.testing.assert_close(
                    loss, torch.tensor(expected_loss * BINARY_ADVANTAGE)
                )
                loss.backward()
                torch.testing.assert_close(
                    current.grad.flatten(),
                    torch.tensor(expected_grad) * BINARY_ADVANTAGE,
                    atol=0,
                    rtol=1e-5,
                )

    def test_unrepresentable_reference_kl_raises_before_backward(self):
        current = torch.tensor([[[-101.0], [-1.0]]], requires_grad=True)
        with self.assertRaisesRegex(AssertionError, "reference KL"):
            grpo_loss(
                current,
                current.detach(),
                torch.zeros(1, 2),
                torch.ones_like(current, dtype=torch.bool),
                ref_log_probs=torch.full_like(current, -1.0),
                kl_beta=0.01,
            )

    def test_invalid_hyperparameters_are_rejected(self):
        current = torch.full((1, 2, 1), -1.0, requires_grad=True)
        invalid_values = {
            "eps": [0.0, -1.0, math.nan, math.inf],
            "clip_eps": [-0.1, 1.0, math.nan, math.inf],
            "kl_beta": [-0.1, math.nan, math.inf],
        }
        for parameter, values in invalid_values.items():
            for value in values:
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(AssertionError, parameter):
                        grpo_loss(
                            current,
                            current.detach(),
                            torch.zeros(1, 2),
                            torch.ones_like(current, dtype=torch.bool),
                            ref_log_probs=current.detach(),
                            **{parameter: value},
                        )

    def test_nonfinite_unmasked_log_probabilities_are_rejected(self):
        for name in ("curr_log_probs", "old_log_probs", "ref_log_probs"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(input=name, value=value):
                    inputs = {
                        key: torch.full((1, 2, 1), -1.0)
                        for key in ("curr_log_probs", "old_log_probs", "ref_log_probs")
                    }
                    inputs[name][0, 0, 0] = value
                    inputs["curr_log_probs"].requires_grad_()
                    with self.assertRaisesRegex(AssertionError, name):
                        grpo_loss(
                            **inputs,
                            rewards=torch.tensor([[1.0, 0.0]]),
                            response_mask=torch.ones((1, 2, 1), dtype=torch.bool),
                            kl_beta=0.1,
                        )

    def test_padding_is_excluded_and_answers_have_equal_weight(self):
        current = torch.tensor(
            [[[math.log(0.275), float("nan"), float("inf")], [math.log(0.225)] * 3]],
            requires_grad=True,
        )
        old = torch.full_like(current, math.log(0.25))
        old[0, 0, 1:] = -torch.inf
        mask = torch.tensor([[[True, False, False], [True, True, True]]])
        loss = grpo_loss(
            current,
            old,
            torch.tensor([[1.0, 0.0]]),
            mask,
            ref_log_probs=current.detach().clone(),
            kl_beta=0.4,
        )

        # Equal answer weights: -(1.1 - 0.9)/2 times the normalized advantage.
        torch.testing.assert_close(loss, torch.tensor(-0.1 * BINARY_ADVANTAGE))
        loss.backward()
        expected_grad = (
            torch.tensor([[[-0.55, 0.0, 0.0], [0.15, 0.15, 0.15]]]) * BINARY_ADVANTAGE
        )
        torch.testing.assert_close(current.grad, expected_grad)

    def test_kl_uses_the_frozen_reference_and_reaches_only_current_policy(self):
        current = torch.tensor(
            [[[math.log(0.25)], [math.log(0.5)]]], requires_grad=True
        )
        reference = torch.tensor(
            [[[math.log(0.5)], [math.log(0.25)]]], requires_grad=True
        )
        old = torch.full_like(current, math.log(0.1), requires_grad=True)
        rewards = torch.zeros(1, 2, requires_grad=True)
        loss = grpo_loss(
            current,
            old,
            rewards,
            torch.ones_like(current, dtype=torch.bool),
            ref_log_probs=reference,
            kl_beta=0.4,
        )

        # Equal rewards isolate KL: 0.4 * ((1-ln(2)) + (ln(2)-0.5)) / 2.
        torch.testing.assert_close(loss, torch.tensor(0.1))
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[[-0.2], [0.1]]]))
        self.assertIsNone(reference.grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(rewards.grad)


if __name__ == "__main__":
    unittest.main()
