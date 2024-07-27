require_relative "lib/torchtext/version"

Gem::Specification.new do |spec|
  spec.name          = "torchtext"
  spec.version       = TorchText::VERSION
  spec.summary       = "Data loaders and abstractions for text and NLP"
  spec.homepage      = "https://github.com/ankane/torchtext-ruby"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 3.1"

  spec.add_dependency "torch-rb", ">= 0.13"
end
