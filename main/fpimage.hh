#ifndef _SEEGNIFY_FP_IMAGE_H_
#define _SEEGNIFY_FP_IMAGE_H_

#include "types.hh"

namespace seegnify {

class FPImage
{
public:

  enum Interpolation
  {
    INTERPOLATE_NEAREST = 1,
    INTERPOLATE_BILINEAR = 2
  };

  FPImage()
  {
    init();
  }

  FPImage(uint32_t rows, uint32_t cols, uint8_t channels = 1)
  {
    init(rows, cols, channels);
  }

  FPImage(FPImage&& other)
  {
    init();

    *this = std::move(other);
  }

  ~FPImage()
  {
    clear();
  }

  FPImage& operator=(FPImage&& other)
  {
    if (this == &other) return *this;

    clear();

    _rows = other._rows;
    _cols = other._cols;
    _channels = other._channels;
    _data = other._data;

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

  FPImage crop (uint32_t row, uint32_t col, uint32_t rows, uint32_t cols) const;

  FPImage scale (uint32_t rows, uint32_t cols, Interpolation interp = INTERPOLATE_NEAREST) const;

  FPImage norm(DTYPE range = 1.0) const;

protected:
  FPImage scale_nearest (uint32_t rows, uint32_t cols) const;
  FPImage scale_bilinear (uint32_t rows, uint32_t cols) const;

  void init()
  {
    _data = nullptr;
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
  }

  void clear()
  {
    delete [] _data;
    init();
  }

private:
  DTYPE* _data;
  uint32_t _rows;
  uint32_t _cols;
  uint8_t _channels;
};

} /* namespace */

#endif /* _SEEGNIFY_FP_IMAGE_H_ */
