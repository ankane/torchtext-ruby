module TorchText
  module NN
    class ScaledDotProduct < Torch::NN::Module
      def initialize(dropout: 0.0, batch_first: false)
        super()
        @dropout = dropout
        @batch_first = batch_first
      end

      def forward(query, key, value, attn_mask: nil, bias_k: nil, bias_v: nil)
        if @batch_first
          query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)
        end

        if !bias_k.nil? && !bias_v.nil?
          unless key.size(-1) == bias_k.size(-1) && key.size(-2) == bias_k.size(-2) && bias_k.size(-3) == 1
            raise "Shape of bias_k is not supported"
          end
          unless value.size(-1) == bias_v.size(-1) && value.size(-2) == bias_v.size(-2) && bias_v.size(-3) == 1
            raise "Shape of bias_v is not supported"
          end
          key = Torch.cat([key, bias_k])
          value = Torch.cat([value, bias_v])
          if !attn_mask.nil?
            attn_mask = Torch::NN::Functional.pad(attn_mask, [0, 1])
          end
        end

        tgt_len, head_dim = query.size(-3), query.size(-1)
        unless query.size(-1) == key.size(-1) && key.size(-1) == value.size(-1)
          raise "The feature dim of query, key, value must be equal."
        end
        unless key.size() == value.size()
          raise "Shape of key, value must match"
        end
        src_len = key.size(-3)
        batch_heads = [query.size(-2), key.size(-2)].max

        # Scale query
        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        query = query * (head_dim.to_f ** -0.5)
        if !attn_mask.nil?
          if attn_mask.dim() != 3
            raise RuntimeError, "attn_mask must be a 3D tensor."
          end
          if (attn_mask.size(-1) != src_len) || (attn_mask.size(-2) != tgt_len) || (attn_mask.size(-3) != 1 && attn_mask.size(-3) != batch_heads)
            raise RuntimeError, "The size of the attn_mask is not correct."
          end
          if attn_mask.dtype != :bool
            raise RuntimeError, "Only bool tensor is supported for attn_mask"
          end
        end

        # Dot product of q, k
        attn_output_weights = Torch.matmul(query, key.transpose(-2, -1))
        if !attn_mask.nil?
          # TODO confirm last argument
          attn_output_weights.masked_fill!(attn_mask, -1e8, nil)
        end
        attn_output_weights = Torch::NN::Functional.softmax(attn_output_weights, dim: -1)
        attn_output_weights = Torch::NN::Functional.dropout(attn_output_weights, p: @dropout, training: @training)
        attn_output = Torch.matmul(attn_output_weights, value)

        if @batch_first
          [attn_output, attn_output_weights]
        else
          [attn_output.transpose(-3, -2), attn_output_weights]
        end
      end
    end
  end
end
