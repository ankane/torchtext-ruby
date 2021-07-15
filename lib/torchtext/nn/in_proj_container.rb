module TorchText
  module NN
    class InProjContainer < Torch::NN::Module
      def initialize(query_proj, key_proj, value_proj)
        super()
        @query_proj = query_proj
        @key_proj = key_proj
        @value_proj = value_proj
      end

      def forward(query, key, value)
        [@query_proj.call(query), @key_proj.call(key), @value_proj.call(value)]
      end
    end
  end
end
