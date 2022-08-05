#ifndef _SEEGNIFY_IMAGE_FP_H_
#define _SEEGNIFY_IMAGE_FP_H_

#include "types.hh"

namespace seegnify {

class ImageFP
{
public:

  enum Status
  {
    STATUS_OK = 0,
    STATUS_ERROR,
    STATUS_INVALID_FILE,
    STATUS_HEADER_NOT_INITIALIZED,
    STATUS_FILE_NOT_OPENED
  };

  enum Interpolation
  {
    INTERPOLATE_NEAREST = 1,
    INTERPOLATE_BILINEAR = 2
  };

  ImageFP()
  {
    init();
  }

  ImageFP(uint32_t rows, uint32_t cols, uint8_t channels = 1)
  {
    init(rows, cols, channels);
  }

  ImageFP(DTYPE* data, uint32_t rows, uint32_t cols, uint8_t channels = 1)
  {
    init(data, rows, cols, channels);
  }

  ImageFP(ImageFP&& other)
  {
    init();

    *this = std::move(other);
  }

  ~ImageFP()
  {
    clear();
  }

  ImageFP& operator=(ImageFP&& other)
  {
    if (this == &other) return *this;

    clear();

    _rows = other._rows;
    _cols = other._cols;
    _channels = other._channels;
    _data = other._data;
    _mydata = other._mydata;

    other.init();
  }

  DTYPE *data() { return _data; }

  const DTYPE *data() const { return _data; }

  uint32_t size() const { return _rows * _cols * _channels; }

  int32_t rows() const { return _rows; }

  int32_t cols() const { return _cols; }

  uint8_t channels() const { return _channels; }

  void set (uint32_t row, uint32_t col, uint8_t channel, DTYPE value);

  DTYPE get (uint32_t row, uint32_t col, uint8_t channel) const;

  Status load(const std::string& filename);

  Status save(const std::string& filename) const;

  ImageFP crop (uint32_t row, uint32_t col, uint32_t rows, uint32_t cols) const;

  ImageFP scale (uint32_t rows, uint32_t cols, Interpolation interp = INTERPOLATE_NEAREST) const;

  ImageFP norm(DTYPE range = 1.0) const;

protected:
  ImageFP scale_nearest (uint32_t rows, uint32_t cols) const;
  ImageFP scale_bilinear (uint32_t rows, uint32_t cols) const;

  void init()
  {
    _data = nullptr;
    _mydata = nullptr;
    _rows = 0;
    _cols = 0;
    _channels = 0;
  }

  void init(uint32_t rows, uint32_t cols, uint8_t channels)
  {
    _rows = rows;
    _cols = cols;
    _channels = channels;
    _data = new DTYPE[rows * cols * channels];
    _mydata = _data;
  }

  void init(DTYPE* data, uint32_t rows, uint32_t cols, uint8_t channels)
  {
    _rows = rows;
    _cols = cols;
    _channels = channels;
    _data = data;
    _mydata = nullptr;
  }

  void clear()
  {
    delete [] _mydata;
    init();
  }

private:
  DTYPE* _data;
  DTYPE* _mydata;
  uint32_t _rows;
  uint32_t _cols;
  uint8_t _channels;
};

} /* namespace */

#endif /* _SEEGNIFY_IMAGE_FP_H_ */
