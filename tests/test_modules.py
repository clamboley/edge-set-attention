import pytest
import torch

from src.ESA.model import ESA, MLP, AttentionBlock, ESAConfig, PoolingModule, SelfAttention


@pytest.fixture
def mock_config() -> ESAConfig:
    """Return a mock ESAConfig object."""
    return ESAConfig(
        layers="SMSMPS",
        n_embd=32,
        n_head=4,
        dropout=0.1,
        bias=True,
        seeds=4,
        pool_layers=2,
    )


def test_mlp_forward(mock_config: ESAConfig) -> None:
    """Test MLP forward pass with random input."""
    mlp = MLP(mock_config)
    x = torch.randn(8, 32)  # batch_size=8, n_embd=32
    out = mlp(x)
    assert out.shape == x.shape, "MLP output shape mismatch"
    assert not torch.isnan(out).any(), "MLP output contains NaNs"


def test_self_attention_invalid_config() -> None:
    """Test SelfAttention with invalid config."""
    config = ESAConfig(n_embd=65, n_head=8)
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        SelfAttention(config)


def test_self_attention_forward(mock_config: ESAConfig) -> None:
    """Test SelfAttention forward pass without mask."""
    self_attn = SelfAttention(mock_config)
    x = torch.randn(8, 16, 32)  # batch_size=8, seq_length=16, n_embd=32

    out = self_attn(x)

    assert out.shape == x.shape, "SelfAttention output shape mismatch (no mask)"
    assert not torch.isnan(out).any(), "SelfAttention output contains NaNs (no mask)"


def test_masked_attention_forward(mock_config: ESAConfig) -> None:
    """Test SelfAttention forward pass with a mask."""
    self_attn = SelfAttention(mock_config)
    x = torch.randn(8, 16, 32)  # batch_size=8, seq_length=16, n_embd=32
    mask = torch.ones(8, 16, 16)

    out = self_attn(x, mask=mask)

    assert out.shape == x.shape, "SelfAttention output shape mismatch (with mask)"
    assert not torch.isnan(out).any(), "SelfAttention output contains NaNs (with mask)"


def test_self_attention_block_forward(mock_config: ESAConfig) -> None:
    """Test AttentionBlock forward pass without mask."""
    attn_block = AttentionBlock(mock_config)
    x = torch.randn(8, 16, 32)  # batch_size=8, num_edges=16, n_embd=32

    out = attn_block(x)
    assert out.shape == x.shape, "AttentionBlock output shape mismatch (no mask)"
    assert not torch.isnan(out).any(), "AttentionBlock output contains NaNs (no mask)"


def test_masked_attention_block_forward(mock_config: ESAConfig) -> None:
    """Test AttentionBlock forward pass with a mask.."""
    attn_block = AttentionBlock(mock_config)
    x = torch.randn(8, 16, 32)  # batch_size=8, num_edges=16, n_embd=32
    mask = torch.ones(8, 16, 16)

    out = attn_block(x, mask)
    assert out.shape == x.shape, "AttentionBlock output shape mismatch (with mask)"
    assert not torch.isnan(out).any(), "AttentionBlock output contains NaNs (with mask)"


def test_pooling_module_forward(mock_config: ESAConfig) -> None:
    """Test PoolingModule forward pass."""
    pooling = PoolingModule(mock_config) # With 4 seed vectors
    z = torch.randn(8, 16, 32)  # batch_size=8, num_edges=16, n_embd=32

    out = pooling(z)
    assert out.shape == (8, 4, 32), "PoolingModule output shape mismatch"
    assert not torch.isnan(out).any(), "PoolingModule output contains NaNs"


def test_esa_initialization(mock_config: ESAConfig) -> None:
    """Test ESA model initialization."""
    model = ESA(mock_config)

    # Check that all blocks are initialized correctly
    ab_blocks = model.transformer["attn_block"]
    out_blocks = model.transformer["out_attn_block"]
    assert len(ab_blocks) + len(out_blocks) + 1 == len(mock_config.layers), "Wrong number of blocks"
    assert all(isinstance(block, AttentionBlock) for block in ab_blocks), "Wrong type in AB block"
    assert all(isinstance(block, AttentionBlock) for block in out_blocks), "Wrong type in out block"
    assert isinstance(model.transformer["pool_mod"], PoolingModule), "PoolingModule not initialized"
    assert not any(p.isnan().any() for p in model.parameters()), "Parameters contain NaNs"


def test_esa_forward_pass(mock_config: ESAConfig) -> None:
    """Test ESA model forward pass."""
    model = ESA(mock_config)

    x = torch.randn(8, 16, 32)  # batch_size=8, num_edges=16, n_embd=32
    mask = torch.ones(8, 16, 16)

    out = model(x, mask)

    assert out.shape == (8, 32), "Output shape mismatch"
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_esa_invalid_layers() -> None:
    """Test ESA model with invalid layers string."""
    for layers in [
        "XMP",     # Invalid: Starts with invalid character.
        "MPP",     # Invalid: Multiple P's.
        "SSMM",    # Invalid: No P.
        "MSPMS",   # Invalid: M in out blocks.
    ]:
        config = ESAConfig(layers=layers)
        with pytest.raises(ValueError, match="Unsupported layers configuration."):
            ESA(config)
