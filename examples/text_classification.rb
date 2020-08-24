# ported from PyTorch Tutorials
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# Copyright (c) 2017 PyTorch contributors, 2020 Andrew Kane
# BSD 3-Clause License

require "torch"
require "torchtext"

ngrams = 2

train_dataset, test_dataset = TorchText::Datasets::AG_NEWS.load(root: ".data", ngrams: ngrams)
batch_size = 16
device = Torch.device(Torch::CUDA.available? ? "cuda" : "cpu")

class TextSentiment < Torch::NN::Module
  def initialize(vocab_size, embed_dim, num_class)
    super()
    @embedding = Torch::NN::EmbeddingBag.new(vocab_size, embed_dim, sparse: true)
    @fc = Torch::NN::Linear.new(embed_dim, num_class)
    init_weights
  end

  def init_weights
    initrange = 0.5
    @embedding.weight.data.uniform!(-initrange, initrange)
    @fc.weight.data.uniform!(-initrange, initrange)
    @fc.bias.data.zero!
  end

  def forward(text, offsets)
    embedded = @embedding.call(text, offsets: offsets)
    @fc.call(embedded)
  end
end

vocab_size = train_dataset.vocab.length
embed_dim = 32
nun_class = train_dataset.labels.length
model = TextSentiment.new(vocab_size, embed_dim, nun_class).to(device)

criterion = Torch::NN::CrossEntropyLoss.new.to(device)
optimizer = Torch::Optim::SGD.new(model.parameters, lr: 4.0)
scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.9)

generate_batch = lambda do |batch|
  label = Torch.tensor(batch.map { |entry| entry[0] })
  text = batch.map { |entry| entry[1] }
  offsets = [0] + text.map { |entry| entry.size }

  # Torch::Tensor.cumsum returns the cumulative sum
  # of elements in the dimension dim.
  offsets = Torch.tensor(offsets[0..-2]).cumsum(0)
  text = Torch.cat(text)
  [text, offsets, label]
end

train_func = lambda do |sub_train_|
  # Train the model
  train_loss = 0
  train_acc = 0
  data = Torch::Utils::Data::DataLoader.new(sub_train_, batch_size: batch_size, shuffle: true, collate_fn: generate_batch)
  data.each_with_index do |(text, offsets, cls), i|
    optimizer.zero_grad
    text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
    output = model.call(text, offsets)
    loss = criterion.call(output, cls)
    train_loss += loss.item
    loss.backward
    optimizer.step
    train_acc += output.argmax(1).eq(cls).sum.item
  end

  # Adjust the learning rate
  scheduler.step

  [train_loss / sub_train_.length, train_acc / sub_train_.length.to_f]
end

test = lambda do |data_|
  loss = 0
  acc = 0
  data = Torch::Utils::Data::DataLoader.new(data_, batch_size: batch_size, collate_fn: generate_batch)
  data.each do |text, offsets, cls|
    text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
    Torch.no_grad do
      output = model.call(text, offsets)
      loss = criterion.call(output, cls)
      loss += loss.item
      acc += output.argmax(1).eq(cls).sum.item
    end
  end

  [loss / data_.length, acc / data_.length.to_f]
end

n_epochs = 5
min_valid_loss = Float::INFINITY

train_len = (train_dataset.length * 0.95).to_i
sub_train_, sub_valid_ = Torch::Utils::Data.random_split(train_dataset, [train_len, train_dataset.length - train_len])

n_epochs.times do |epoch|
  start_time = Time.now
  train_loss, train_acc = train_func.call(sub_train_)
  valid_loss, valid_acc = test.call(sub_valid_)

  secs = Time.now - start_time
  mins = secs / 60
  secs = secs % 60

  puts "Epoch: %d | time in %d minutes, %d seconds" % [epoch + 1, mins, secs]
  puts "\tLoss: %.4f (train)\t|\tAcc: %.1f%% (train)" % [train_loss, train_acc * 100]
  puts "\tLoss: %.4f (valid)\t|\tAcc: %.1f%% (valid)" % [valid_loss, valid_acc * 100]
end

puts "Checking the results of test dataset..."
test_loss, test_acc = test.call(test_dataset)
puts "\tLoss: %.4f (test)\t|\tAcc: %.1f%% (test)" % [test_loss, test_acc * 100]

ag_news_label = {1 => "World",
                 2 => "Sports",
                 3 => "Business",
                 4 => "Sci/Tec"}

def predict(text, model, vocab, ngrams)
  tokenizer = TorchText::Data::Utils.tokenizer("basic_english")
  Torch.no_grad do
    text = Torch.tensor(TorchText::Data::Utils.ngrams_iterator(tokenizer.call(text), ngrams).map { |token| vocab[token] })
    output = model.call(text, Torch.tensor([0]))
    output.argmax(1).item + 1
  end
end

ex_text_str = <<~EOS
  MEMPHIS, Tenn. – Four days ago, Jon Rahm was
  enduring the season’s worst weather conditions on Sunday at The
  Open on his way to a closing 75 at Royal Portrush, which
  considering the wind and the rain was a respectable showing.
  Thursday’s first round at the WGC-FedEx St. Jude Invitational
  was another story. With temperatures in the mid-80s and hardly any
  wind, the Spaniard was 13 strokes better in a flawless round.
  Thanks to his best putting performance on the PGA Tour, Rahm
  finished with an 8-under 62 for a three-stroke lead, which
  was even more impressive considering he’d never played the
  front nine at TPC Southwind.
EOS

vocab = train_dataset.vocab
model = model.to("cpu")

puts "This is a %s news" % ag_news_label[predict(ex_text_str, model, vocab, 2)]
