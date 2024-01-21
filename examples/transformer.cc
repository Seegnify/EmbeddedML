/*
 * Copyright 2020-2024 Greg Padiasek. All Rights Reserved.
 */

#include "main/training.hh"
#include "main/optimizer.hh"
#include "main/storage.hh"

#include "transformer.hh"

#define NUM_LAYERS 2    // number of encoder/decoder layers
#define NUM_HEADS 4     // numberof multiheads
#define EMB_SIZE 16     // size of embeddings
#define SEQ_SIZE 32     // max seqeunce size
#define FF_SIZE 64      // size of feed forward hidden layer
#define DROPOUT 0.0     // dropout probability

#define SRC_TOKENS 128  // input vocabulary size
#define TGT_TOKENS 128  // output vocabulary size
#define BOS_TOKEN 1     // begining of sequence token
#define EOS_TOKEN 2     // end of sequence token
#define PAD_TOKEN 3     // padding token (marks positions to ignore)

/////////////////////////////////////
// training instance implementation
/////////////////////////////////////
class TransformerClient : public Training
{
public:
  TransformerClient(int worker) : Training(worker)
  {
    std::cout << "TRANSFORMER training " << worker << std::endl;
    
    // load data
    _train_data.push_back(
      {
        "Hi, how is it going?",
        "Czesc, co u ciebie?"
      }
    );
    _train_data.push_back(
      {
        "When is your next train?",
        "Kiedy jest twoj nastepny pociag?",
      }
    );
    _train_data.push_back(
      {
        "I like chocolate.",
        "Lubie czekolade.",
      }
    );

    // init train batch sequence
    for (int i=0; i<_train_data.size(); i++) _train_batch.push_back(i);

    Graph& g = graph();

    // model
    _model = new Transformer(g,
      SRC_TOKENS, TGT_TOKENS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
      NUM_LAYERS, NUM_HEADS, EMB_SIZE, FF_SIZE, SEQ_SIZE, DROPOUT);
    g.keep(_model);

    // predicted output
    auto y = g.new_rowwise(*_model, SEQ_SIZE, TGT_TOKENS,
      [&](Function& row) { return g.new_log_softmax(row); });

    // expected output
    _y_hat = new SequenceMask(g, PAD_TOKEN, SEQ_SIZE);
    g.keep(_y_hat);
    
    // loss
    _tgt_size = g.new_constant(1,1); // target sequence size
    _loss = g.new_sum(- *_y_hat * *y); // cross entropy
    _loss = &(*_loss / *_tgt_size / TGT_TOKENS); // normalized loss

    // optimizer
    //_optimizer = new Adam(g.variables(), 0.0001);
    _optimizer = new SGD(g.variables(), 0.001, 0.1);

    // counters
    _batch = 0;
  }

  ~TransformerClient()
  {
    delete _optimizer;
  }

  // tokenize and pad source sequence
  std::vector<int> source_tokens(const std::string& text)
  {
    std::vector<int> tokens(SEQ_SIZE, PAD_TOKEN);

    auto size = std::min<int>(text.size(), SEQ_SIZE);
    for (int i=0; i<size; i++) tokens[i] = text[i];

    return tokens;
  }

  // tokenize and pad target sequence
  std::vector<int> target_tokens(const std::string& text)
  {
    std::vector<int> tokens(SEQ_SIZE, PAD_TOKEN);

    tokens[0] = BOS_TOKEN; // BOS as first token

    auto size = std::min<int>(text.size(), SEQ_SIZE - 1); // 1 for BOS
    for (int i=0; i<size; i++) tokens[i + 1] = text[i]; // 1 for BOS

    return tokens;
  }

  // generate output mask for training loss
  Tensor output_mask(const std::vector<int>& tgt)
  {
    Tensor mask = Tensor::Zero(SEQ_SIZE, TGT_TOKENS);

    auto it = std::find(tgt.begin(), tgt.end(), PAD_TOKEN);
    size_t tgt_size = ((it != tgt.end()) ? it - tgt.begin() : tgt.size()) - 1;

    // one-hot token
    for (int i=0; i<tgt_size; i++)
    {
      mask(i, tgt[i + 1]) = 1; // i + 1 for BOS
    }

    // one-hot EOS
    mask(tgt_size, EOS_TOKEN) = 1;
    return mask;
  }

  virtual void batch_train()
  {
    // restore references
    auto& g = graph();
    auto& opt = *_optimizer;

    int batch_size = _train_data.size();
    _batch++;

    // sample training data
    g.random().shuffle(_train_batch.begin(), _train_batch.end());

    // batch train
    for (int i=0; i<batch_size; i++)
    {
      g.recache();

      // training sample
      auto& x = _train_data[_train_batch[i]];
      auto src_x = source_tokens(x.first);
      auto tgt_x = target_tokens(x.second);

      // training mask
      _y_hat->value() = output_mask(tgt_x);
      
      if (worker() == 0)
      {
        std::cout << "=== training step ===" << std::endl;
        std::cout << "batch id:" << i << std::endl;
        std::cout << "sample id:" << _train_batch[i] << std::endl;
        std::cout << "source:" << x.first << std::endl;
        for (auto t: src_x) std::cout << "src token:" << t << std::endl;
        std::cout << "target:" << x.second << std::endl;
        for (auto t: tgt_x) std::cout << "tgt token:" << t << std::endl;
        std::cout << "y_hat (output mask):" << std::endl;
        std::cout << _y_hat->forward() << std::endl;
        exit(1);
      }
      
      // generator
      //Tensor y = _model->forward(src_x, tgt_x);
      //_model->backward(image);
    }

    // update weights
    //opt.update();
    //g.zero_grad();

    int valid_step = 100;
    if (_batch % valid_step == 0 and worker() == 0)
    {
      std::cout << "validation success "
                << validate(g)
                << std::endl;
    }
  }

  float validate(Graph& g)
  {
    float success = 0;

    int size = 0;

    for (int i=0; i<size; i++)
    {
      g.recache();

      // generator
      //success += _model->forward(image, label, false);
    }

    return success / size;
  }

private:
  std::vector<std::pair<std::string, std::string>> _train_data;
  std::vector<int> _train_batch;
  Transformer *_model;
  Optimizer *_optimizer;
  Constant* _tgt_size;
  SequenceMask* _y_hat;
  Function* _loss;
  int _batch;
};

///////////////////////////////////
extern "C" { // export C signatures
///////////////////////////////////

DLL_EXPORT Training* create(int idx)
{
  return new TransformerClient(idx);
}

DLL_EXPORT void destroy(Training* ptr)
{
  delete (TransformerClient*)ptr;
}

///////////////////////////////////
} // export C signatures
///////////////////////////////////

