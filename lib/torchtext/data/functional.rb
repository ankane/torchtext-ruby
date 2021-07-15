module TorchText
  module Data
    module Functional
      class << self
        def simple_space_split(iterator)
          iterator.map(&:split)
        end
      end
    end
  end
end
