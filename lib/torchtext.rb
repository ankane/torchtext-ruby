# dependencies
require "torch"

# stdlib
require "csv"
require "fileutils"
require "rubygems/package"
require "set"

# modules
require "torchtext/data/utils"
require "torchtext/datasets/text_classification"
require "torchtext/datasets/text_classification_dataset"
require "torchtext/vocab"
require "torchtext/version"

module TorchText
  class Error < StandardError; end
end
