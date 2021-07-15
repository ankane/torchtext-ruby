module TorchText
  module Data
    module Metrics
      class << self
        def bleu_score(candidate_corpus, references_corpus, max_n: 4, weights: [0.25] * 4)
          unless max_n == weights.length
            raise "Length of the \"weights\" list has be equal to max_n"
          end
          unless candidate_corpus.length == references_corpus.length
            raise "The length of candidate and reference corpus should be the same"
          end

          clipped_counts = Torch.zeros(max_n)
          total_counts = Torch.zeros(max_n)
          weights = Torch.tensor(weights)

          candidate_len = 0.0
          refs_len = 0.0

          candidate_corpus.zip(references_corpus) do |candidate, refs|
            candidate_len += candidate.length

            # Get the length of the reference that's closest in length to the candidate
            refs_len_list = refs.map { |ref| ref.length.to_f }
            refs_len += refs_len_list.min_by { |x| (candidate.length - x).abs }

            reference_counters = compute_ngram_counter(refs[0], max_n)
            refs[1..-1].each do |ref|
              reference_counters = reference_counters.merge(compute_ngram_counter(ref, max_n)) { |_, v1, v2| v1 > v2 ? v1 : v2 }
            end

            candidate_counter = compute_ngram_counter(candidate, max_n)

            shared_keys = candidate_counter.keys & reference_counters.keys
            clipped_counter = candidate_counter.slice(*shared_keys).merge(reference_counters.slice(*shared_keys)) { |_, v1, v2| v1 < v2 ? v1 : v2 }

            clipped_counter.each_key do |ngram|
              clipped_counts[ngram.length - 1] += clipped_counter[ngram]
            end

            candidate_counter.each_key do |ngram|
              total_counts[ngram.length - 1] += candidate_counter[ngram]
            end
          end

          if clipped_counts.to_a.min == 0
            0.0
          else
            pn = clipped_counts / total_counts
            log_pn = weights * Torch.log(pn)
            score = Torch.exp(log_pn.sum)

            bp = Math.exp([1 - refs_len / candidate_len, 0].min)

            bp * score.item
          end
        end

        private

        def compute_ngram_counter(tokens, max_n)
          raise "Failed assert" unless max_n > 0
          Hash[TorchText::Data::Utils.ngrams_iterator(tokens, max_n).map { |x| x.split(" ") }.group_by { |v| v }.map { |k, v| [k, v.size] }]
        end
      end
    end
  end
end
