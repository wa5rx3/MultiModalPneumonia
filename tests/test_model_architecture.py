# Model architecture tests — DenseNet backbone patched to weights=None so no download needed.
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torchvision.models as _real_tv_models

# Save before patching so the helper always calls the real function.
_real_densenet121 = _real_tv_models.densenet121


def _densenet_no_weights(*args, **kwargs):
    return _real_densenet121(weights=None)


class TestTabularMLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("src.models.multimodal_model.models.densenet121",
                   side_effect=_densenet_no_weights):
            from src.models.multimodal_model import TabularMLP
        cls.TabularMLP = TabularMLP

    def _mlp(self, in_dim: int):
        return self.TabularMLP(input_dim=in_dim)

    def test_output_shape(self):
        mlp = self._mlp(21)
        x = torch.randn(8, 21)
        out = mlp(x)
        self.assertEqual(out.shape, (8, 128))

    def test_various_input_dims(self):
        for dim in (5, 21, 58, 100):
            mlp = self._mlp(dim)
            x = torch.randn(4, dim)
            out = mlp(x)
            self.assertEqual(out.shape, (4, 128),
                             f"Failed for input_dim={dim}")

    def test_output_dtype_float32(self):
        mlp = self._mlp(21)
        out = mlp(torch.randn(2, 21))
        self.assertEqual(out.dtype, torch.float32)

    def test_grad_flows_through_mlp(self):
        mlp = self._mlp(21)
        x = torch.randn(4, 21, requires_grad=True)
        mlp(x).sum().backward()
        self.assertIsNotNone(x.grad)


class TestConcatModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("src.models.multimodal_model.models.densenet121",
                   side_effect=_densenet_no_weights):
            from src.models.multimodal_model import MultimodalPneumoniaModel
            cls.model = MultimodalPneumoniaModel(tabular_input_dim=21)

    def test_instantiation(self):
        self.assertIsNotNone(self.model)

    def test_forward_pass_shape(self):
        img = torch.randn(2, 3, 224, 224)
        tab = torch.randn(2, 21)
        logits = self.model(img, tab)
        self.assertEqual(logits.shape, (2, 1))

    def test_output_is_logit_not_prob(self):
        """Model must not apply a final sigmoid — output is raw logit."""
        last_layer = list(self.model.fusion_head.modules())[-1]
        self.assertNotIsInstance(last_layer, nn.Sigmoid)

    def test_backbone_freeze(self):
        self.model.freeze_image_backbone()
        for p in self.model.image_backbone.parameters():
            self.assertFalse(p.requires_grad)

    def test_backbone_unfreeze(self):
        self.model.freeze_image_backbone()
        self.model.unfreeze_image_backbone()
        for p in self.model.image_backbone.parameters():
            self.assertTrue(p.requires_grad)

    def test_trainable_param_count_changes(self):
        self.model.unfreeze_image_backbone()
        unfrozen = self.model.total_trainable_parameters()
        self.model.freeze_image_backbone()
        frozen = self.model.total_trainable_parameters()
        self.assertLess(frozen, unfrozen)

    def test_image_backbone_trainable_parameters(self):
        self.model.unfreeze_image_backbone()
        bb_params = self.model.image_backbone_trainable_parameters()
        self.assertGreater(bb_params, 0)

    def test_forward_output_deterministic_in_eval(self):
        """Same input → same output in eval mode."""
        self.model.eval()
        img = torch.randn(1, 3, 224, 224)
        tab = torch.randn(1, 21)
        with torch.no_grad():
            out1 = self.model(img, tab)
            out2 = self.model(img, tab)
        self.assertTrue(torch.allclose(out1, out2))
        self.model.train()


class TestAttnModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("src.models.multimodal_model.models.densenet121",
                   side_effect=_densenet_no_weights):
            from src.models.multimodal_model_attn import MultimodalPneumoniaModelAttn
            from src.models.multimodal_model import MultimodalPneumoniaModel
            cls.model = MultimodalPneumoniaModelAttn(tabular_input_dim=21)
            cls.concat_model = MultimodalPneumoniaModel(tabular_input_dim=21)

    def test_instantiation(self):
        self.assertIsNotNone(self.model)

    def test_forward_pass_shape(self):
        img = torch.randn(2, 3, 224, 224)
        tab = torch.randn(2, 21)
        logits = self.model(img, tab)
        self.assertEqual(logits.shape, (2, 1))

    def test_output_matches_concat_shape(self):
        """Both fusion models must return the same [B, 1] shape."""
        img = torch.randn(4, 3, 224, 224)
        tab = torch.randn(4, 21)
        attn_out = self.model(img, tab)
        concat_out = self.concat_model(img, tab)
        self.assertEqual(attn_out.shape, concat_out.shape)

    def test_backbone_freeze_attn(self):
        self.model.freeze_image_backbone()
        for p in self.model.image_backbone.parameters():
            self.assertFalse(p.requires_grad)
        self.model.unfreeze_image_backbone()

    def test_cls_token_is_learnable(self):
        """CLS token should be a learnable parameter."""
        self.assertTrue(self.model.cls_token.requires_grad)

    def test_grad_flows_end_to_end(self):
        img = torch.randn(2, 3, 224, 224)
        tab = torch.randn(2, 21)
        logits = self.model(img, tab)
        logits.sum().backward()
        self.assertIsNotNone(self.model.cls_token.grad)

    def test_different_tabular_dims_work(self):
        """Attention model should accept different tabular input dims."""
        with patch("src.models.multimodal_model.models.densenet121",
                   side_effect=_densenet_no_weights):
            from src.models.multimodal_model_attn import MultimodalPneumoniaModelAttn
            m = MultimodalPneumoniaModelAttn(tabular_input_dim=58)
        img = torch.randn(2, 3, 224, 224)
        tab = torch.randn(2, 58)
        out = m(img, tab)
        self.assertEqual(out.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
