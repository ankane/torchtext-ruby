require_relative "test_helper"

class NNTest < Minitest::Test
  def test_multihead_attention_container
    embed_dim = 10
    num_heads = 5
    bsz = 64

    in_proj_container = TorchText::NN::InProjContainer.new(
      Torch::NN::Linear.new(embed_dim, embed_dim),
      Torch::NN::Linear.new(embed_dim, embed_dim),
      Torch::NN::Linear.new(embed_dim, embed_dim)
    )

    mha = TorchText::NN::MultiheadAttentionContainer.new(
      num_heads,
      in_proj_container,
      TorchText::NN::ScaledDotProduct.new,
      Torch::NN::Linear.new(embed_dim, embed_dim)
    )

    query = Torch.rand([21, bsz, embed_dim])
    key = value = Torch.rand([16, bsz, embed_dim])
    attn_output, attn_weights = mha.call(query, key, value)
    assert_equal [21, 64, 10], attn_output.shape
  end
end
