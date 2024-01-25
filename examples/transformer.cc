/*
 * Copyright 2020-2024 Greg Padiasek. All Rights Reserved.
 */

#include "main/training.hh"
#include "main/optimizer.hh"
#include "main/storage.hh"

#include "transformer.hh"

/* Standard Transfomer options */
#define NUM_LAYERS 6    // number of encoder/decoder layers
#define NUM_HEADS 8     // numberof multiheads
#define EMB_SIZE 512    // size of embeddings
#define SEQ_SIZE 100    // max seqeunce size
#define FF_SIZE 2048    // size of feed forward hidden layer
#define DROPOUT 0.1     // dropout probability

/* Test Transfomer options */
/*
#define NUM_LAYERS 2    // number of encoder/decoder layers
#define NUM_HEADS 2     // numberof multiheads
#define EMB_SIZE 64     // size of embeddings
#define SEQ_SIZE 48     // max seqeunce size
#define FF_SIZE 16      // size of feed forward hidden layer
#define DROPOUT 0.0     // dropout probability
*/

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
    _model = new Transformer(g, SRC_TOKENS, TGT_TOKENS, PAD_TOKEN,
      NUM_LAYERS, NUM_HEADS, EMB_SIZE, FF_SIZE, SEQ_SIZE, DROPOUT);
    g.keep(_model);

    // predicted output
    auto y = g.new_rowwise(*_model, SEQ_SIZE, TGT_TOKENS,
      [&](Function& row) { return g.new_log_softmax(row); });
    g.name(y, "softmax_y");

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

  // determine actual sequence size
  int sequence_size(const std::vector<int>& sequence)
  {
    auto it = std::find(sequence.begin(), sequence.end(), PAD_TOKEN);

    if (it == sequence.end()) return sequence.size();

    // limit seqence size to max sequence size
    return std::min<int>(it - sequence.begin(), SEQ_SIZE);
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

    // select training batch
    g.random().shuffle(_train_batch.begin(), _train_batch.end());

    // batch train
    for (int i=0; i<batch_size; i++)
    {
      g.recache();

      // training sample
      auto& x = _train_data[_train_batch[i]];
      auto src_x = source_tokens(x.first);
      auto tgt_x = target_tokens(x.second);

      // loss mask
      _y_hat->value() = output_mask(tgt_x);

      // loss normalization
      _tgt_size->value()(0) = sequence_size(tgt_x);
      
      // set model inputs
      auto& y = _model->forward(src_x, tgt_x);

      // traning loss
      auto& loss = _loss->forward();
      if (worker() == 0)
      {
        std::cout << "time:" << time_to_string(time_now())
                  << " loss:" << loss
                  << std::endl;
      }

      // backward pass from loss
      g.backward(*_loss, Tensor::Ones(1,1));
    }

    // update weights
    opt.update();
    g.zero_grad();

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

