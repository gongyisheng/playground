"""Numerical contracts; run from ml with python -m unittest discover -s rl."""

import math
import unittest

import torch
from rl.score_centering import (
    score_centering_loss,
    score_centering_token_loss,
    top_k_top_p_distribution,
)


class ScoreCenteringTests(unittest.TestCase):
    def test_sampling_filters_and_defaults(self):
        cases = [
            # Nucleus mass is measured AFTER top-k renormalization.
            (
                [0.4, 0.25, 0.2, 0.1, 0.05],
                {"top_p": 0.7, "top_k": 3},
                [8 / 13, 5 / 13, 0, 0, 0],
            ),
            # Keep at least one token, even when it alone exceeds top-p.
            ([0.96, 0.02, 0.015, 0.005], {}, [1, 0, 0, 0]),
            ([0.5, 0.3, 0.2], {"top_p": 1.0, "top_k": 0}, [0.5, 0.3, 0.2]),
            # Defaults cap at 20: weights 6..25 sum to 310. The nucleus
            # keeps 8..25 (mass 297/310), including the crossing token 8.
            (list(range(1, 26)), {}, [0] * 7 + [v / 297 for v in range(8, 26)]),
        ]
        for weights, kwargs, expected in cases:
            with self.subTest(kwargs=kwargs, vocab_size=len(weights)):
                logits = (
                    torch.tensor(weights, dtype=torch.float32).log().requires_grad_()
                )
                ids, probabilities = top_k_top_p_distribution(logits, **kwargs)
                dense = torch.zeros_like(logits).scatter(-1, ids, probabilities)
                torch.testing.assert_close(
                    dense, torch.tensor(expected, dtype=torch.float32)
                )
                self.assertFalse(probabilities.requires_grad)

    def test_loss_has_centered_gradient_and_frozen_rollout_inputs(self):
        # Let c=q*w and m=sum(c). Independently derived logit gradients are
        # A * (c - w_y * one_hot(y) + (w_y - m) * p), with frozen weights.
        cases = [
            (
                [0.2, 0.3, 0.5],
                [0.6, 0.3, 0.1],
                0,
                2.0,
                {},
                0.6 * math.log(1.5)
                + 0.2 * math.log(2.5)
                - 2 * sum(p * math.log(p) for p in (0.2, 0.3, 0.5)),
                [-0.8, 0.6, 0.2],
            ),
            (
                [0.7, 0.1, 0.2],
                [0.6, 0.3, 0.1],
                1,
                -3.0,
                {},
                1.8 * math.log(1 / 7)
                + 0.3 * math.log(0.5)
                + 3 * sum(p * math.log(p) for p in (0.7, 0.1, 0.2)),
                [-1.8, 2.1, -0.3],
            ),
            # On-policy, centering leaves the ordinary policy gradient intact.
            (
                [0.2, 0.3, 0.5],
                [0.2, 0.3, 0.5],
                2,
                1.0,
                {},
                -math.log(0.5),
                [0.2, 0.3, -0.5],
            ),
            # A deterministic sampler has no score covariance or learning signal.
            ([0.2, 0.3, 0.5], [0.6, 0.3, 0.1], 0, 1.0, {"top_k": 1}, 0.0, [0.0] * 3),
            # Ratios [0.2, 0.75, 6]: TIS caps the sampled ratio at 2.
            (
                [0.1, 0.3, 0.6],
                [0.5, 0.4, 0.1],
                2,
                1.0,
                {"importance_sampling": "tis"},
                -2.4 * math.log(0.6),
                [0.24, 0.72, -0.96],
            ),
            # MIS drops both out-of-range head weights; only 0.75 survives.
            (
                [0.1, 0.3, 0.6],
                [0.5, 0.4, 0.1],
                1,
                1.0,
                {"importance_sampling": "mis"},
                -0.1 * math.log(0.1) - 0.75 * math.log(0.3) - 0.6 * math.log(0.6),
                [0.045, -0.315, 0.27],
            ),
            # Even a masked sampled token has a centering correction.
            (
                [0.1, 0.3, 0.6],
                [0.5, 0.4, 0.1],
                2,
                -2.0,
                {"importance_sampling": "mis"},
                0.2 * math.log(0.1) + 1.2 * math.log(0.6),
                [0.06, -0.42, 0.36],
            ),
            # Custom TIS cap: weights [0.2, 0.75, 1], not [0.2, 0.75, 2].
            (
                [0.1, 0.3, 0.6],
                [0.5, 0.4, 0.1],
                2,
                1.0,
                {"importance_sampling": "tis", "tis_cap": 1.0},
                -1.5 * math.log(0.6),
                [0.15, 0.45, -0.6],
            ),
            # Inclusive custom MIS bounds: retain a ratio exactly equal to 2.
            (
                [0.5, 0.5],
                [0.25, 0.75],
                0,
                1.0,
                {"importance_sampling": "mis", "mis_min": 2.0, "mis_max": 2.0},
                -1.5 * math.log(0.5),
                [-0.75, 0.75],
            ),
            # Raw ratios exceed float32 range, but clipped/masked weights don't.
            (
                [0.5, 0.5],
                [1.0, 1e-40],
                1,
                1.0,
                {"importance_sampling": "tis"},
                -2.5 * math.log(0.5),
                [1.25, -1.25],
            ),
            (
                [0.5, 0.5],
                [1.0, 1e-40],
                1,
                1.0,
                {"importance_sampling": "mis"},
                -0.5 * math.log(0.5),
                [0.25, -0.25],
            ),
        ]
        for p, q, token, advantage, options, expected_loss, expected_grad in cases:
            with self.subTest(token=token, advantage=advantage, options=options):
                current = torch.tensor([[p]]).log().requires_grad_()
                sampler = torch.tensor([[q]]).log().requires_grad_()
                advantages = torch.tensor([advantage], requires_grad=True)
                loss = score_centering_loss(
                    current,
                    sampler,
                    torch.tensor([[token]]),
                    advantages,
                    torch.ones(1, 1, dtype=torch.bool),
                    top_p=1.0,
                    **{"top_k": 0, **options},
                )
                torch.testing.assert_close(loss, torch.tensor(expected_loss))
                loss.backward()
                torch.testing.assert_close(
                    current.grad.flatten(), torch.tensor(expected_grad)
                )
                self.assertIsNone(sampler.grad)
                self.assertIsNone(advantages.grad)

    def test_weighted_centering_cancels_drift_with_truncated_sampling(self):
        for method in ("tis", "mis"):
            with self.subTest(method=method):
                current = (
                    torch.tensor([[[0.1, 0.8, 0.05, 0.05]]]).log().requires_grad_()
                )
                sampler = torch.tensor([[[0.5, 0.25, 0.125, 0.125]]]).log()
                # top-k=3 then top-p=0.8 gives q=[2/3, 1/3, 0, 0], leaving
                # a zero-probability nucleus entry AND a token outside top-k.
                # Enumerate every possible sampled token with constant reward.
                losses = [
                    score_centering_loss(
                        current,
                        sampler,
                        torch.tensor([[token]]),
                        torch.ones(1),
                        torch.ones(1, 1, dtype=torch.bool),
                        top_p=0.8,
                        top_k=3,
                        importance_sampling=method,
                    )
                    for token in (0, 1)
                ]
                expected_loss = (2 * losses[0] + losses[1]) / 3
                # Miles's residual form has a different scalar surrogate, but
                # the same zero expected gradient for a constant reward.
                entropy = -(math.log(1 / 9) / 9 + 8 * math.log(8 / 9) / 9)
                torch.testing.assert_close(
                    expected_loss, torch.tensor(entropy), atol=1e-6, rtol=0
                )
                expected_loss.backward()
                torch.testing.assert_close(
                    current.grad, torch.zeros_like(current), atol=1e-6, rtol=0
                )

    def test_invalid_importance_sampling_configuration_is_rejected(self):
        cases = [({"importance_sampling": "typo"}, "importance_sampling")]
        cases += [
            ({"importance_sampling": "tis", "tis_cap": value}, "tis_cap")
            for value in (0.0, -1.0, math.inf, math.nan)
        ]
        cases += [
            ({"importance_sampling": "mis", "mis_min": value}, "mis_min")
            for value in (0.0, -1.0, 6.0, math.inf, math.nan)
        ]
        cases += [
            ({"importance_sampling": "mis", "mis_max": value}, "mis_max")
            for value in (0.0, -1.0, math.inf, math.nan)
        ]
        for options, error in cases:
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(AssertionError, error),
            ):
                score_centering_loss(
                    torch.zeros(1, 1, 2),
                    torch.zeros(1, 1, 2),
                    torch.zeros(1, 1, dtype=torch.long),
                    torch.ones(1),
                    torch.ones(1, 1, dtype=torch.bool),
                    **options,
                )

    def test_padding_and_response_averaging_in_mixed_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                current = (
                    torch.tensor([0.2, 0.3, 0.5]).log().repeat(1, 2, 3, 1).to(dtype)
                )
                current[0, 0, 1] = torch.nan
                current[0, 0, 2] = torch.inf
                current.requires_grad_()
                sampler = torch.tensor([0.6, 0.3, 0.1]).log().repeat(1, 2, 3, 1)
                sampler[0, 0, 1:] = torch.nan
                mask = torch.tensor([[[True, False, False], [True, True, True]]])
                loss = score_centering_loss(
                    current,
                    sampler,
                    torch.tensor([[[0, -100, -100], [2, 2, 2]]]),
                    torch.tensor([[2.0, -1.0]]),
                    mask,
                    top_p=1.0,
                    top_k=0,
                )
                expected_loss = (
                    0.3 * math.log(1.5)
                    + 0.4 * math.log(2.5)
                    + 0.15 * math.log(5 / 3)
                    - 0.5 * sum(p * math.log(p) for p in (0.2, 0.3, 0.5))
                )
                torch.testing.assert_close(
                    loss,
                    torch.tensor(expected_loss),
                    rtol=0,
                    atol=1e-6 if dtype == torch.float32 else 0.005,
                )
                loss.backward()
                expected_grad = torch.tensor(
                    [
                        [
                            [[-0.4, 0.3, 0.1], [0.0] * 3, [0.0] * 3],
                            [[-0.1, -0.05, 0.15]] * 3,
                        ]
                    ],
                    dtype=dtype,
                )
                torch.testing.assert_close(current.grad, expected_grad)

    def test_filtered_scores_replay_support_and_temperature(self):
        # p on the saved support is [1/9, 8/9], q is [2/3, 1/3].
        # Large trainer logits OUTSIDE that support must have no influence.
        cases = [
            (
                "none",
                (5 * math.log(1 / 9) - 14 * math.log(8 / 9)) / 9,
                [2 / 3, -2 / 3, 0, 0],
            ),
            ("tis", -20 * math.log(8 / 9) / 9, [20 / 81, -20 / 81, 0, 0]),
            (
                "mis",
                -8 * math.log(8 / 9) / 3 - math.log(1 / 9) / 9,
                [16 / 81, -16 / 81, 0, 0],
            ),
        ]
        for mode, expected_loss, expected_grad in cases:
            for temperature in (0.7, 1.0):
                for filtered in (False, True):
                    with self.subTest(
                        mode=mode, temperature=temperature, filtered=filtered
                    ):
                        current = (
                            torch.tensor(
                                [[[0.1, 0.8, 4.0, 9.0]]], dtype=torch.float64
                            ).log()
                            * temperature
                        ).requires_grad_()
                        q = (
                            [2 / 3, 1 / 3, 0, 0]
                            if filtered
                            else [0.5, 0.25, 0.125, 0.125]
                        )
                        sampler = torch.tensor([[q]], dtype=torch.float64).log()
                        loss = score_centering_loss(
                            current,
                            sampler,
                            torch.tensor([[1]]),
                            torch.ones(1),
                            torch.ones(1, 1, dtype=torch.bool),
                            top_p=1.0 if filtered else 0.8,
                            top_k=0 if filtered else 3,
                            importance_sampling=mode,
                            temperature=temperature,
                        )
                        torch.testing.assert_close(
                            loss, torch.tensor(expected_loss, dtype=torch.float64)
                        )
                        loss.backward()
                        torch.testing.assert_close(
                            current.grad.flatten(),
                            torch.tensor(expected_grad, dtype=torch.float64)
                            / temperature,
                        )

    def test_partial_head_matches_miles_with_sample_outside_head(self):
        # Reference values from Miles 86304536's real estimator, with head IDs
        # [0,1], sampled ID 3, and recorded q[3]=0.05 (different from q_hat[3]).
        cases = [
            ("none", -0.44016625995059844, [0.5, 0.25, 3 / 28, -6 / 7]),
            ("tis", 1.6748254375236746, [0.22, 0.44, 201 / 350, -216 / 175]),
            ("mis", 0.23025850929940458, [-0.09, 0.02, 0.03, 0.04]),
        ]
        for mode, expected_loss, expected_grad in cases:
            with self.subTest(mode=mode):
                logits = (
                    torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
                    .log()
                    .requires_grad_()
                )
                logp = logits.log_softmax(-1)
                # Include an unused candidate slot to check padding.
                head = torch.cat((logp[:, :2], torch.full((1, 1), torch.nan)), -1)
                q_head = (
                    torch.tensor([[0.5, 0.25, torch.nan]], dtype=torch.float64)
                    .log()
                    .requires_grad_()
                )
                q_sample = (
                    torch.tensor([0.05], dtype=torch.float64).log().requires_grad_()
                )
                advantages = torch.ones(1, dtype=torch.float64, requires_grad=True)
                loss = score_centering_token_loss(
                    logp[:, 3],
                    head,
                    q_sample,
                    q_head,
                    torch.tensor([[True, True, False]]),
                    advantages,
                    importance_sampling=mode,
                ).sum()
                torch.testing.assert_close(
                    loss, torch.tensor(expected_loss, dtype=torch.float64)
                )
                loss.backward()
                torch.testing.assert_close(
                    logits.grad.flatten(),
                    torch.tensor(expected_grad, dtype=torch.float64),
                )
                self.assertIsNone(q_head.grad)
                self.assertIsNone(q_sample.grad)
                self.assertIsNone(advantages.grad)

    def test_token_reduction_matches_miles_recipe(self):
        # On-policy residual=0; one response has one token, the other has three.
        logits = (
            torch.tensor([0.25, 0.75], dtype=torch.float64)
            .log()
            .repeat(2, 3, 1)
            .requires_grad_()
        )
        ids = torch.tensor([[0, -100, -100], [1, 1, 1]])
        mask = ids != -100
        for reduction, expected in (
            ("response_mean", -(math.log(0.25) + math.log(0.75)) / 2),
            ("token_mean", -(math.log(0.25) + 3 * math.log(0.75)) / 4),
        ):
            with self.subTest(reduction=reduction):
                loss = score_centering_loss(
                    logits,
                    logits.detach(),
                    ids,
                    torch.ones(2),
                    mask,
                    top_p=1.0,
                    top_k=0,
                    reduction=reduction,
                )
                torch.testing.assert_close(
                    loss, torch.tensor(expected, dtype=torch.float64)
                )


if __name__ == "__main__":
    unittest.main()
