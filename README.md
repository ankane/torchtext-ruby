# TorchText Ruby

:fire: Data loaders and abstractions for text and NLP - for Ruby

[![Build Status](https://github.com/ankane/torchtext-ruby/workflows/build/badge.svg?branch=master)](https://github.com/ankane/torchtext-ruby/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'torchtext'
```

## Getting Started

This library follows the [Python API](https://pytorch.org/text/). Many methods and options are missing at the moment. PRs welcome!

## Examples

Text classification

- [PyTorch tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [Ruby code](examples/text_classification.rb)

## Datasets

Load a dataset

```ruby
train_dataset, test_dataset = TorchText::Datasets::AG_NEWS.load(root: ".data", ngrams: 2)
```

Supported datasets are:

- [AG_NEWS](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

## Data Utils

Supports:

- tokenizer
- ngrams_iterator

## Data Metrics

Compute the BLEU score

```ruby
candidate_corpus = [["My", "full", "pytorch", "test"], ["Another", "Sentence"]]
references_corpus = [[["My", "full", "pytorch", "test"], ["Completely", "Different"]], [["No", "Match"]]]
TorchText::Data::Metrics.bleu_score(candidate_corpus, references_corpus)
```

## NN

Supports:

- InProjContainer
- MultiheadAttentionContainer
- ScaledDotProduct

## Vocab

Supports:

- Vocab

## Disclaimer

This library downloads and prepares public datasets. We don’t host any datasets. Be sure to adhere to the license for each dataset.

If you’re a dataset owner and wish to update any details or remove it from this project, let us know.

## History

View the [changelog](https://github.com/ankane/torchtext-ruby/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/torchtext-ruby/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/torchtext-ruby/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/torchtext-ruby.git
cd torchtext-ruby
bundle install
bundle exec rake test
```
