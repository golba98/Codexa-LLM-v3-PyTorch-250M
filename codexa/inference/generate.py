import torch
from tokenizers import Tokenizer
from torch import Tensor

from codexa.architecture.model import Codexa
from codexa.config import Config
from codexa.tokenizer.bpe import BOS, EOS, CodexaTokenizer
from codexa.training.checkpoint import load
from codexa.training.trainer import choose_device


def filter_logits(logits: Tensor, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    scores = logits.clone().float()
    if top_k:
        threshold = scores.topk(min(top_k, scores.shape[-1])).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        sorted_scores, indices = scores.sort(descending=True)
        remove = sorted_scores.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scores.scatter_(-1, indices, sorted_scores.masked_fill(remove, -torch.inf))
    return scores


@torch.inference_mode()
def generate(
    model: Codexa,
    prompt: list[int],
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    seed: int = 42,
    eos: int = EOS,
    precision: str = "fp32",
    use_cache: bool = True,
    max_sequence: int | None = None,
) -> list[int]:
    if max_tokens < 0 or temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling settings")
    if precision not in ("fp32", "bf16", "auto"):
        raise ValueError("Invalid inference precision")
    device = next(model.parameters()).device
    bf16 = precision != "fp32" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = list(prompt) or [BOS]
    limit = max_sequence if max_sequence is not None else model.config.context
    if not 0 < limit <= model.config.context or len(ids) + max_tokens > limit:
        raise ValueError(
            "Prompt plus generation budget exceeds context; shorten prompt/budget or explicitly load a larger context profile"
        )
    cache = None
    training = model.training
    model.eval()
    try:
        for _ in range(max_tokens):
            x = torch.tensor([ids if cache is None else ids[-1:]], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                if use_cache:
                    result, cache = model.forward_cached(
                        x, cache, capacity=len(ids) + max_tokens if cache is None else None
                    )
                    logits = result[0, -1].float()
                else:
                    logits = model(x, last_logits_only=True)[0, -1].float()
            if greedy:
                token = int(logits.argmax())
            else:
                scores = filter_logits(logits / temperature, top_k, top_p)
                token = int(torch.multinomial(scores.softmax(-1), 1, generator=generator))
            ids.append(token)
            if token == eos:
                break
        return ids
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "Inference OOM: shorten prompt/budget, select BF16 when supported, or use a smaller model; no context was truncated"
        ) from exc
    finally:
        model.train(training)


def from_checkpoint(path: str, prompt: str, device: str = "auto", **kwargs: object) -> str:
    state = load(path)
    model = Codexa(Config.from_dict(state["config"]).model).to(choose_device(device))
    model.load_state_dict(state["model"])
    tokenizer = CodexaTokenizer(Tokenizer.from_str(state["tokenizer_json"]))
    config = Config.from_dict(state["config"])
    options = {k: v for k, v in kwargs.items() if v is not None}
    for k in ("max_sequence", "max_tokens", "use_cache", "precision"):
        options.setdefault(k, getattr(config.inference, k))
    return tokenizer.decode(generate(model, [BOS, *tokenizer.encode(prompt)], **options))  # type: ignore[arg-type]
