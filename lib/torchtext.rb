# dependencies
require "torch"

# stdlib
require "csv"
require "fileutils"
require "rubygems/package"
require "set"

# modules
require_relative "torchtext/data/functional"
require_relative "torchtext/data/metrics"
require_relative "torchtext/data/utils"
require_relative "torchtext/datasets/text_classification"
require_relative "torchtext/datasets/text_classification_dataset"
require_relative "torchtext/nn/in_proj_container"
require_relative "torchtext/nn/multihead_attention_container"
require_relative "torchtext/nn/scaled_dot_product"
require_relative "torchtext/vocab"
require_relative "torchtext/version"

module TorchText
  class Error < StandardError; end
end
