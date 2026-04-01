"""Classifier wrappers for text, image, and multimodal classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification


class BertClassifier(nn.Module):
    """Wrapper around AutoModelForSequenceClassification."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        # Suppress expected UNEXPECTED/MISSING key warnings when loading a
        # base checkpoint into a sequence classification head.
        orig_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        transformers.logging.set_verbosity(orig_verbosity)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass returning logits."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We compute loss ourselves for weighted CE
        )
        return outputs.logits

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask):
        """Return softmax probabilities (no gradient)."""
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)


class ImageClassifier(nn.Module):
    """CLIP ViT vision encoder with a linear classification head."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        from transformers import CLIPVisionModel

        orig_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        transformers.logging.set_verbosity(orig_verbosity)

        hidden_size = self.vision_model.config.hidden_size  # 768 for clip-vit-base-patch32
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        """Forward pass returning logits."""
        outputs = self.vision_model(pixel_values=pixel_values)
        cls_embedding = outputs.pooler_output  # (batch, 768)
        return self.classifier(cls_embedding)

    @torch.no_grad()
    def predict_proba(self, pixel_values):
        """Return softmax probabilities (no gradient)."""
        logits = self.forward(pixel_values)
        return F.softmax(logits, dim=-1)


class MultimodalClassifier(nn.Module):
    """Late fusion: BERTweet text encoder + CLIP ViT image encoder.

    Concatenates [CLS] embeddings from both encoders, then classifies
    through a single linear head.
    """

    def __init__(self, text_model_name: str, image_model_name: str, num_labels: int):
        super().__init__()
        from transformers import CLIPVisionModel

        orig_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(image_model_name)
        transformers.logging.set_verbosity(orig_verbosity)

        text_hidden = self.text_model.config.hidden_size    # 768 for bertweet-base
        image_hidden = self.vision_model.config.hidden_size  # 768 for clip-vit-base-patch32
        self.classifier = nn.Linear(text_hidden + image_hidden, num_labels)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """Forward pass returning logits."""
        # Text branch
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0, :]  # [CLS] token (batch, 768)

        # Image branch
        image_out = self.vision_model(pixel_values=pixel_values)
        image_cls = image_out.pooler_output  # (batch, 768)

        # Late fusion
        fused = torch.cat([text_cls, image_cls], dim=-1)  # (batch, 1536)
        return self.classifier(fused)

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask, pixel_values):
        """Return softmax probabilities (no gradient)."""
        logits = self.forward(input_ids, attention_mask, pixel_values)
        return F.softmax(logits, dim=-1)


def create_fresh_model(config) -> nn.Module:
    """Factory to create a fresh classifier from config, dispatched by modality."""
    if config.modality == "image_only":
        return ImageClassifier(
            model_name=config.image_model_name,
            num_labels=config.num_labels,
        )
    elif config.modality == "text_image":
        return MultimodalClassifier(
            text_model_name=config.model_name,
            image_model_name=config.image_model_name,
            num_labels=config.num_labels,
        )
    else:  # text_only (default)
        return BertClassifier(
            model_name=config.model_name,
            num_labels=config.num_labels,
        )
