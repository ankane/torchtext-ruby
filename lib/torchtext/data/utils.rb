module TorchText
  module Data
    module Utils
      def tokenizer(tokenizer, language: "en")
        return method(:split_tokenizer) if tokenizer.nil?

        if tokenizer == "basic_english"
          if language != "en"
            raise ArgumentError, "Basic normalization is only available for English(en)"
          end
          return method(:basic_english_normalize)
        end

        raise "Not implemented yet"
      end

      def ngrams_iterator(token_list, ngrams)
        return enum_for(:ngrams_iterator, token_list, ngrams) unless block_given?

        get_ngrams = lambda do |n|
          (token_list.size - n + 1).times.map { |i| token_list[i...(i + n)] }
        end

        token_list.each do |x|
          yield x
        end
        2.upto(ngrams) do |n|
          get_ngrams.call(n).each do |x|
            yield x.join(" ")
          end
        end
      end

      private

      def split_tokenizer(x)
        x.split
      end

      _patterns = [%r{\'}, %r{\"}, %r{\.}, %r{<br \/>}, %r{,}, %r{\(}, %r{\)}, %r{\!}, %r{\?}, %r{\;}, %r{\:}, %r{\s+}]
      _replacements = [" \'  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]

      PATTERNS_DICT = _patterns.zip(_replacements)

      def basic_english_normalize(line)
        line = line.downcase

        PATTERNS_DICT.each do |pattern_re, replaced_str|
          line.sub!(pattern_re, replaced_str)
        end
        line.split
      end

      extend self
    end

    # TODO only tokenizer method
    extend Utils
  end
end
