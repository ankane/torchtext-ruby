# dependencies
require "torch"

# stdlib
require "csv"
require "fileutils"
require "rubygems/package"
require "set"

# modules
require "torchtext/data/utils"
require "torchtext/data/metrics"
require "torchtext/datasets/text_classification"
require "torchtext/datasets/text_classification_dataset"
require "torchtext/nn/in_proj_container"
require "torchtext/nn/multihead_attention_container"
require "torchtext/nn/scaled_dot_product"
require "torchtext/vocab"
require "torchtext/version"

module TorchText
  class Error < StandardError; end
end
