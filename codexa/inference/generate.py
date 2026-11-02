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
) -> list[int]:
    if max_tokens < 0 or temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling settings")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = list(prompt) or [BOS]
    training = model.training
    model.eval()
    try:
        for _ in range(max_tokens):
            x = torch.tensor([ids[-model.config.context :]], device=device)
            logits = model(x)[0, -1].float()
            if greedy:
                token = int(logits.argmax())
            else:
                scores = filter_logits(logits / temperature, top_k, top_p)
                token = int(torch.multinomial(scores.softmax(-1), 1, generator=generator))
            ids.append(token)
            if token == eos:
                break
        return ids
    finally:
        model.train(training)


def from_checkpoint(path: str, prompt: str, device: str = "auto", **kwargs: object) -> str:
    state = load(path)
    model = Codexa(Config.from_dict(state["config"]).model).to(choose_device(device))
    model.load_state_dict(state["model"])
    tokenizer = CodexaTokenizer(Tokenizer.from_str(state["tokenizer_json"]))
    return tokenizer.decode(generate(model, [BOS, *tokenizer.encode(prompt)], **kwargs))  # type: ignore[arg-type]
