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

  enum Channel
  {
    CHANNEL_BLUE = 0,
    CHANNEL_GREEN = 1,
    CHANNEL_RED = 2
  };

  Image()
  {
    init();
  }

  Image(uint32_t rows, uint32_t cols, uint8_t channels = 3)
  {
    init(rows, cols, channels);
  }

  Image(uint8_t* data, uint32_t rows, uint32_t cols, uint8_t channels = 3)
  {
    init(data, rows, cols, channels);
  }

  Image(Image&& other)
  {
    init();

    *this = std::move(other);
  }

  ~Image()
  {
    clear();
  }

  Image& operator=(Image&& other)
  {
    if (this == &other) return *this;

    clear();

    _rows = other._rows;
    _cols = other._cols;
    _channels = other._channels;
    _data = other._data;
    _mydata = other._mydata;

    other.init();

    return *this;
  }

  uint8_t *data() { return _data; }

  const uint8_t *data() const { return _data; }

  uint32_t size() const { return _rows * _cols * _channels; }

  int32_t rows() const { return _rows; }

  int32_t cols() const { return _cols; }

  uint8_t channels() const { return _channels; }

  void set (uint32_t row, uint32_t col, uint8_t channel, uint8_t val);

  uint8_t get (uint32_t row, uint32_t col, uint8_t channel) const;

  void set (uint32_t row, uint32_t col, uint8_t r, uint8_t g, uint8_t b);

  uint8_t red (uint32_t row, uint32_t col) const
  {
    return get(row, col, CHANNEL_RED);
  }

  uint8_t green (uint32_t row, uint32_t col) const
  {
    return get(row, col, CHANNEL_GREEN);
  }

  uint8_t blue (uint32_t row, uint32_t col) const
  {
    return get(row, col, CHANNEL_BLUE);
  }

  Status load (const std::string& filename);

  Status save (const std::string& filename) const;

  Image crop (uint32_t row, uint32_t col, uint32_t rows, uint32_t cols) const;

  Image scale (uint32_t rows, uint32_t cols, Interpolation interp = INTERPOLATE_NEAREST) const;

  Image norm() const;

protected:
  Image scale_nearest (uint32_t rows, uint32_t cols) const;
  Image scale_bilinear (uint32_t rows, uint32_t cols) const;

  void write_row (uint32_t row, std::ofstream& f) const;
  void read_row (uint32_t row, std::ifstream& f);

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
    _data = new uint8_t[rows * cols * channels];
    _mydata = _data;
  }

  void init(uint8_t* data, uint32_t rows, uint32_t cols, uint8_t channels)
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
  uint8_t* _data;
  uint8_t* _mydata;
  uint32_t _rows;
  uint32_t _cols;
  uint8_t _channels;
};

} /* namespace */

#endif /* _SEEGNIFY_IMAGE_H_ */
