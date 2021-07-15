module TorchText
  module NN
    class MultiheadAttentionContainer < Torch::NN::Module
      def initialize(nhead, in_proj_container, attention_layer, out_proj, batch_first: false)
        super()
        @nhead = nhead
        @in_proj_container = in_proj_container
        @attention_layer = attention_layer
        @out_proj = out_proj
        @batch_first = batch_first
      end

      def forward(query, key, value, attn_mask: nil, bias_k: nil, bias_v: nil)
        if @batch_first
          query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)
        end

        tgt_len, src_len, bsz, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)
        q, k, v = @in_proj_container.call(query, key, value)
        unless q.size(-1) % @nhead == 0
          raise "query's embed_dim must be divisible by the number of heads"
        end
        head_dim = q.size(-1).div(@nhead)
        q = q.reshape(tgt_len, bsz * @nhead, head_dim)

        unless k.size(-1) % @nhead == 0
          raise "key's embed_dim must be divisible by the number of heads"
        end
        head_dim = k.size(-1).div(@nhead)
        k = k.reshape(src_len, bsz * @nhead, head_dim)

        unless v.size(-1) % @nhead == 0
          raise "value's embed_dim must be divisible by the number of heads"
        end
        head_dim = v.size(-1).div(@nhead)
        v = v.reshape(src_len, bsz * @nhead, head_dim)

        attn_output, attn_output_weights = @attention_layer.call(q, k, v, attn_mask: attn_mask, bias_k: bias_k, bias_v: bias_v)
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        attn_output = @out_proj.call(attn_output)

        if @batch_first
          attn_output = attn_output.transpose(-3, -2)
        end

        [attn_output, attn_output_weights]
      end
    end
  end
end
