/*
 * Copyright 2020-2023 Greg Padiasek. All Rights Reserved.
 */

#include "main/training.hh"
#include "main/optimizer.hh"
#include "main/storage.hh"

#include "transformer.hh"


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
        "I like chockolate.",
        "Lubie czekolade.",
      }
    );

    // create graph
    int NUM_LAYERS = 1;
    int NUM_HEADS = 2;
    int EMB_SIZE = 4;
    int SEQ_SIZE = 5;
    int FF_SIZE = 3;
    DTYPE DROPOUT = 0.0;

    int SRC_TOKENS = 10; // input vocabulary size
    int TGT_TOKENS = 10; // output vocabulary size
    int SOS_TOKEN = 8; // start of sequence token
    int EOS_TOKEN = 9; // end of sequence token
    int PAD_TOKEN = 0; // padding token (marks positions to ignore)

    Graph g;

    _model = new Transformer( g, SRC_TOKENS, TGT_TOKENS, PAD_TOKEN,
      NUM_LAYERS, NUM_HEADS, EMB_SIZE, FF_SIZE, SEQ_SIZE, DROPOUT);

    auto y = g.new_rowwise(*_model, SEQ_SIZE, TGT_TOKENS,
      [&](Function& row) { return g.new_log_softmax(row); });
    
    // loss
    _y_hat = g.new_constant(SEQ_SIZE, TGT_TOKENS); // expected output
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
    delete _model;
    delete _optimizer;
  }
    
  virtual void batch_train()
  {
    // restore references
    auto& g = graph();
    auto& opt = *_optimizer;

    int batch_size = 100;
    _batch++;
    float success = 0;

    // batch train
    for (int i=0; i<batch_size; i++)
    {
      g.recache();
      
      std::vector<int> src_x;
      std::vector<int> tgt_x;
      
      // generator
      Tensor y = _model->forward(src_x, tgt_x);
      //_model->backward(image);
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
  Transformer *_model;
  Optimizer *_optimizer;
  Constant* _tgt_size;
  Constant* _y_hat;
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

