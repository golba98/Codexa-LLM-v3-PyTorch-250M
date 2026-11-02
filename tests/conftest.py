import pytest
import torch

from codexa.data.packing import prepare, read_documents, split_documents
from codexa.tokenizer.bpe import CodexaTokenizer

torch.set_num_threads(1)


@pytest.fixture
def prepared(tmp_path):
    tok_path = tmp_path / "tokenizer.json"
    docs, _ = split_documents(read_documents("examples/corpus"))
    tok = CodexaTokenizer.train(docs, 512)
    tok.save(tok_path)
    data = tmp_path / "data"
    prepare("examples/corpus", str(tok_path), str(data))
    return str(data), str(tok_path), str(tmp_path / "run")
