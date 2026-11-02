from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)


class CodexaTokenizer:
    def __init__(self, backend: Tokenizer):
        self.backend = backend
        if any(backend.token_to_id(s) != i for i, s in enumerate(SPECIALS)):
            raise ValueError("Incompatible special tokens")

    @classmethod
    def train(cls, documents: list[str], vocab_size: int) -> "CodexaTokenizer":
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(
            documents,
            trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=SPECIALS,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False,
            ),
        )
        if backend.get_vocab_size() != vocab_size:
            raise ValueError(
                "Corpus cannot supply requested vocabulary; use more diverse training text"
            )
        return cls(backend)

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "CodexaTokenizer":
        return cls(Tokenizer.from_file(str(path)))
