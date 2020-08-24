require_relative "lib/torchtext/version"

Gem::Specification.new do |spec|
  spec.name          = "torchtext"
  spec.version       = TorchText::VERSION
  spec.summary       = "Data loaders and abstractions for text and NLP"
  spec.homepage      = "https://github.com/ankane/torchtext"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.5"

  spec.add_dependency "torch-rb", ">= 0.3.2"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "minitest", ">= 5"
end
