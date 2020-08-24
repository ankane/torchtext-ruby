module TorchText
  class Vocab
    UNK = "<unk>"

    def initialize(
      counter, max_size: nil, min_freq: 1, specials: ["<unk>", "<pad>"],
      vectors: nil, unk_init: nil, vectors_cache: nil, specials_first: true
    )

      @freqs = counter
      counter = counter.dup
      min_freq = [min_freq, 1].max

      @itos = []
      @unk_index = nil

      if specials_first
        @itos = specials
        # only extend max size if specials are prepended
        max_size += specials.size if max_size
      end

      # frequencies of special tokens are not counted when building vocabulary
      # in frequency order
      specials.each do |tok|
        counter.delete(tok)
      end

      # sort by frequency, then alphabetically
      words_and_frequencies = counter.sort_by { |k, v| [-v, k] }

      words_and_frequencies.each do |word, freq|
        break if freq < min_freq || @itos.length == max_size
        @itos << word
      end

      if specials.include?(UNK)  # hard-coded for now
        unk_index = specials.index(UNK)  # position in list
        # account for ordering of specials, set variable
        @unk_index = specials_first ? unk_index : @itos.length + unk_index
        @stoi = Hash.new(@unk_index)
      else
        @stoi = {}
      end

      if !specials_first
        @itos.concat(specials)
      end

      # stoi is simply a reverse dict for itos
      @itos.each_with_index do |tok, i|
        @stoi[tok] = i
      end

      @vectors = nil
      if !vectors.nil?
        # self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        raise "Not implemented yet"
      else
        raise "Failed assertion" unless unk_init.nil?
        raise "Failed assertion" unless vectors_cache.nil?
      end
    end

    def [](token)
      @stoi.fetch(token, @stoi.fetch(UNK))
    end

    def length
      @itos.length
    end
    alias_method :size, :length

    def self.build_vocab_from_iterator(iterator)
      counter = Hash.new(0)
      i = 0
      iterator.each do |tokens|
        tokens.each do |token|
          counter[token] += 1
        end
        i += 1
        puts "Processed #{i}" if i % 10000 == 0
      end
      Vocab.new(counter)
    end
  end
end
