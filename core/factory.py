from core.text_only_bert import TextOnlyBERT
from core.audio_visual_baseline import AudioVisualBaseline
from core.bottleneck_fusion import BottleneckFusion


def build_model(args):
    """Build one of the three models from parsed args.

    Supported: text_only, audio_visual, bottleneck
    """
    if args.model == "text_only":
        return TextOnlyBERT(
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            freeze_bert=args.freeze_bert,
        )
    elif args.model == "audio_visual":
        return AudioVisualBaseline(
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            audio_input_dim=args.audio_input_dim,
            visual_input_dim=args.visual_input_dim,
        )
    elif args.model == "bottleneck":
        return BottleneckFusion(
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            num_bottleneck_tokens=args.num_bottleneck_tokens,
            dropout=args.dropout,
            freeze_bert=args.freeze_bert,
            use_audio=not args.no_audio,
            use_visual=not args.no_visual,
            audio_input_dim=args.audio_input_dim,
            visual_input_dim=args.visual_input_dim,
        )
    else:
        raise ValueError(
            f"Unknown model: '{args.model}'. Choose from: text_only, audio_visual, bottleneck"
        )
