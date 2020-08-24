require_relative "test_helper"

class DatasetsTest < Minitest::Test
  def test_ag_news
    train_dataset, test_dataset = TorchText::Datasets::AG_NEWS.load(root: root, ngrams: 2)
    assert_equal 120000, train_dataset.size
    assert_equal 7600, test_dataset.size
  end
end
