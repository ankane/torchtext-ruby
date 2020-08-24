module TorchText
  module Datasets
    class TextClassificationDataset < Torch::Utils::Data::Dataset
      attr_reader :labels, :vocab

      def initialize(vocab, data, labels)
        super()
        @data = data
        @labels = labels
        @vocab = vocab
      end

      def [](i)
        @data[i]
      end

      def length
        @data.length
      end
      alias_method :size, :length

      def each
        @data.each do |x|
          yield x
        end
      end
    end
  end
end
