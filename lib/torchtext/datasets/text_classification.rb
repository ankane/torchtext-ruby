module TorchText
  module Datasets
    module TextClassification
      URLS = {
        "AG_NEWS" => "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms"
      }
      PATHS = {
        "AG_NEWS" => "ag_news_csv"
      }
      FILENAMES = {
        "AG_NEWS" => "ag_news_csv.tar.gz"
      }

      class << self
        def ag_news(*args, **kwargs)
          setup_datasets("AG_NEWS", *args, **kwargs)
        end

        private

        def setup_datasets(dataset_name, root: ".data", ngrams: 1, vocab: nil, include_unk: false)
          dataset_tar = download_from_url(URLS[dataset_name], root: root, filename: FILENAMES[dataset_name])
          to_path = extract_archive(dataset_tar)
          extracted_files = Dir["#{to_path}/#{PATHS[dataset_name]}/*"]

          train_csv_path = nil
          test_csv_path = nil
          extracted_files.each do |fname|
            if fname.end_with?("train.csv")
              train_csv_path = fname
            elsif fname.end_with?("test.csv")
              test_csv_path = fname
            end
          end

          if vocab.nil?
            vocab = Vocab.build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
          else
            unless vocab.is_a?(Vocab)
              raise ArgumentError, "Passed vocabulary is not of type Vocab"
            end
          end
          train_data, train_labels = _create_data_from_iterator(vocab, _csv_iterator(train_csv_path, ngrams, yield_cls: true), include_unk)
          test_data, test_labels = _create_data_from_iterator(vocab, _csv_iterator(test_csv_path, ngrams, yield_cls: true), include_unk)
          if (train_labels ^ test_labels).length > 0
            raise ArgumentError, "Training and test labels don't match"
          end

          [
            TextClassificationDataset.new(vocab, train_data, train_labels),
            TextClassificationDataset.new(vocab, test_data, test_labels)
          ]
        end

        def _csv_iterator(data_path, ngrams, yield_cls: false)
          return enum_for(:_csv_iterator, data_path, ngrams, yield_cls: yield_cls) unless block_given?

          tokenizer = Data.tokenizer("basic_english")
          CSV.foreach(data_path) do |row|
            tokens = row[1..-1].join(" ")
            tokens = tokenizer.call(tokens)
            if yield_cls
              yield row[0].to_i - 1, Data::Utils.ngrams_iterator(tokens, ngrams)
            else
              yield Data::Utils.ngrams_iterator(tokens, ngrams)
            end
          end
        end

        def _create_data_from_iterator(vocab, iterator, include_unk)
          data = []
          labels = []
          iterator.each do |cls, tokens|
            if include_unk
              tokens = Torch.tensor(tokens.map { |token| vocab[token] })
            else
              token_ids = tokens.map { |token| vocab[token] }.select { |x| x != Vocab::UNK }
              tokens = Torch.tensor(token_ids)
            end
            data << [cls, tokens]
            labels << cls
          end
          [data, Set.new(labels)]
        end

        # extra filename parameter
        def download_from_url(url, root:, filename:)
          path = File.join(root, filename)
          return path if File.exist?(path)

          FileUtils.mkdir_p(root)

          puts "Downloading #{url}..."
          download_url_to_file(url, path)
        end

        # follows redirects
        def download_url_to_file(url, dst)
          uri = URI(url)
          tmp = nil
          location = nil

          Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
            request = Net::HTTP::Get.new(uri)

            http.request(request) do |response|
              case response
              when Net::HTTPRedirection
                location = response["location"]
              when Net::HTTPSuccess
                tmp = "#{Dir.tmpdir}/#{Time.now.to_f}" # TODO better name
                File.open(tmp, "wb") do |f|
                  response.read_body do |chunk|
                    f.write(chunk)
                  end
                end
              else
                raise Error, "Bad response"
              end
            end
          end

          if location
            download_url_to_file(location, dst)
          else
            FileUtils.mv(tmp, dst)
            dst
          end
        end

        # extract_tar_gz doesn't list files, so just return to_path
        def extract_archive(from_path, to_path: nil, overwrite: nil)
          to_path ||= File.dirname(from_path)

          if from_path.end_with?(".tar.gz") || from_path.end_with?(".tgz")
            File.open(from_path, "rb") do |io|
              Gem::Package.new("").extract_tar_gz(io, to_path)
            end
            return to_path
          end

          raise "We currently only support tar.gz and tgz archives"
        end
      end

      DATASETS = {
        "AG_NEWS" => method(:ag_news)
      }

      LABELS = {
        "AG_NEWS" => {
          0 => "World",
          1 => "Sports",
          2 => "Business",
          3 => "Sci/Tech"
        }
      }
    end

    class AG_NEWS
      def self.load(*args, **kwargs)
        TextClassification.ag_news(*args, **kwargs)
      end
    end
  end
end
