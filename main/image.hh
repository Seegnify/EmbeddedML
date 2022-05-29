#ifndef _SEEGNIFY_IMAGE_H_
#define _SEEGNIFY_IMAGE_H_

#include <string>

namespace seegnify {

class Image
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

  Image()
  {
    _data = nullptr;
    _rows = 0;
    _cols = 0;
    _bits_per_pixel = 0;
  }

  Image(uint32_t rows, uint32_t cols, uint8_t bpp = 24)
  {
    init(rows, cols, bpp);
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

  uint8_t *data() { return _data; }

  int32_t rows() { return _rows; }

  int32_t cols() { return _cols; }

  uint8_t bits_per_pixel() { return _bits_per_pixel; }  

  void set (uint32_t row, uint32_t col, uint8_t r, uint8_t g, uint8_t b);

  uint8_t red (uint32_t row, uint32_t col);

  uint8_t green (uint32_t row, uint32_t col);

  uint8_t blue (uint32_t row, uint32_t col);

  Status load (const std::string& filename);

  Status save (const std::string& filename);

  Image crop (uint32_t row, uint32_t col, uint32_t rows, uint32_t cols);

  Image scale (uint32_t rows, uint32_t cols, Interpolation interp = INTERPOLATE_NEAREST);

protected:
  Image scale_nearest (uint32_t rows, uint32_t cols);
  Image scale_bilinear (uint32_t rows, uint32_t cols);

  void write_row (uint32_t row, std::ofstream& f);
  void read_row (uint32_t row, std::ifstream& f);

  void init(uint32_t rows, uint32_t cols, uint8_t bpp)
  {
    _rows = rows;
    _cols = cols;
    _bits_per_pixel = bpp;
    _data = new uint8_t[rows * cols * bpp / 8];
  }

  void clear()
  {
    delete [] _data;
    _data = nullptr;
    _rows = 0;
    _cols = 0;
    _bits_per_pixel = 0;
  }

private:
  uint8_t* _data;
  uint32_t _rows;
  uint32_t _cols;
  uint8_t _bits_per_pixel;
};

} /* namespace */

#endif /* _SEEGNIFY_IMAGE_H_ */
