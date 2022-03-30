#ifndef _SEEGNIFY_IMAGE_H_
#define _SEEGNIFY_IMAGE_H_

#include <string>

namespace seegnify {

class Image
{
public:

  enum Interpolation
  {
    INTERPOLATE_NEAREST = 1,
    INTERPOLATE_BILINEAR = 2
  };

  Image()
  {
    _data = nullptr;
    _rows = 0;
    _cols = 0;
    _bits_per_pixel = 0;
  }

  Image(uint32_t rows, uint32_t cols, uint8_t bits_per_pixel = 24)
  {
    _rows = rows;
    _cols = cols;
    _bits_per_pixel = bits_per_pixel;
    _data = new uint8_t[rows * cols * bits_per_pixel / 8];
  }

  Image(Image&& other)
  {
    _rows = other._rows;
    _cols = other._cols;
    _bits_per_pixel = other._bits_per_pixel;
    _data = other._data;

    other._data = nullptr;
    other._rows = 0;
    other._cols = 0;
    other._bits_per_pixel = 0;
  }

  ~Image()
  {
    clear();
  }

  void clear()
  {
    delete [] _data;
    _data = nullptr;
    _rows = 0;
    _cols = 0;
    _bits_per_pixel = 0;
  }

  uint8_t *data() { return _data; }

  int32_t rows() { return _rows; }

  int32_t cols() { return _cols; }

  uint8_t bits_per_pixel() { return _bits_per_pixel; }  

  void load(const std::string& path);

  void save(const std::string& path);

  Image crop(int roq, int col, uint32_t rows, uint32_t cols);

  Image scale(uint32_t rows, uint32_t cols,
  Interpolation interp = INTERPOLATE_NEAREST);

protected:
  Image scale_nearest(uint32_t rows, uint32_t cols);
  Image scale_bilinear(uint32_t rows, uint32_t cols);

private:
  uint8_t *_data;
  uint32_t _rows;
  uint32_t _cols;
  uint8_t _bits_per_pixel;
};

} /* namespace */

#endif /* _SEEGNIFY_IMAGE_H_ */
