/*
 * Copyright 2020-2023 Greg Padiasek
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "rlenv.hh"
#include "painter.hh"

namespace seegnify {

// default image settings
#define VIEW_ROWS 16 // default number of view rows
#define VIEW_COLS 16 // default number of view cols
#define CHANNELS 3   // default channles - RGB

// scale range
#define MAX_SCALE 10.0
#define MIN_SCALE 0.1

// rounding routine
#define ROUND(a) ((int)round((a)))

// view-data offset
#define OFFSET_X() ROUND(this->x - this->view_rows/2.0/this->scale)
#define OFFSET_Y() ROUND(this->y - this->view_cols/2.0/this->scale)

// real dimensions
#define DATA_X(ax) ROUND((ax) / this->scale + (OFFSET_X()))
#define DATA_Y(ay) ROUND((ay) / this->scale + (OFFSET_Y()))
#define DATA_L(al) ROUND((al) / this->scale) // data length

// view dimensions
#define VIEW_X(rx) ROUND((rx - (OFFSET_X())) * this->scale)
#define VIEW_Y(ry) ROUND((ry - (OFFSET_Y())) * this->scale)
#define VIEW_L(rl) ROUND((rl) * this->scale) // view length

// view/data array index
#define VIEW_I(x,y,channel) (3*((y)*this->view_cols + (x)) + (channel))
#define DATA_I(x,y,channel) (3*((y)*this->data_cols + (x)) + (channel))

RLEnv::RLEnv()
{
  data = nullptr;

  show_view_frame = false;

  slices = 0;
  data_rows = 0;
  data_cols = 0;
  view_rows = VIEW_ROWS;
  view_cols = VIEW_COLS;

  new_episode();
}

RLEnv::~RLEnv()
{
  clear();
}

////////////////////////// instance RL API /////////////////////////////////

void RLEnv::new_episode()
{
  action_step = 0;
  scale = 1.0;

  slice = 0;
  x = data_cols / 2;
  y = data_rows / 2;
}

void RLEnv::set_data_rgb(uint8_t *data, uint16_t slices, uint16_t rows, uint16_t cols)
{
  // clear data if needed
  if (this->slices != slices || 
      this->data_rows != rows || 
      this->data_cols != cols)
    clear();

  // allocated data if needed  
  int size = CHANNELS * slices * rows * cols;
  if (this->data == nullptr) this->data = new uint8_t[size];
  memcpy(this->data, data, size);

  // update dimenstions
  this->slices = slices;
  this->data_rows = rows;
  this->data_cols = cols;
}

void RLEnv::set_view_size(uint16_t rows, uint16_t cols)
{
  this->view_rows = rows;
  this->view_cols = cols;
}

////////////////////////// instance UI API /////////////////////////////////

void RLEnv::get_data_size(uint16_t& data_rows, uint16_t& data_cols)
{
  data_rows = this->data_rows;
  data_cols = this->data_cols;
}

void RLEnv::get_view_size(uint16_t& data_rows, uint16_t& data_cols)
{
  data_rows = this->view_rows;
  data_cols = this->view_cols;
}

Image RLEnv::get_data_rgb()
{
  // get slice index
  auto slice = ROUND(this->slice);

  // draw data view
  int slice_offset = CHANNELS * slice * data_rows * data_cols;
  Image view(data + slice_offset, data_rows, data_cols, CHANNELS);
  
  // copy data view
  Image img(view.rows(), view.cols(), view.channels());
  memcpy(img.data(), view.data(), view.size());

  // draw agent frame
  draw_agent_frame(img);

  return img;
}

Image RLEnv::get_view_rgb()
{
  // crop data to agent view
  Image real = get_data_rgb();
  auto view = real.crop(DATA_Y(0), DATA_X(0), DATA_L(view_rows), DATA_L(view_cols));

  // zoom to fit agent view
  return view.scale(view_rows, view_cols);
}

std::string RLEnv::get_info()
{
  char buf[128];
  std::ostringstream os;

  auto ac = view_cols;
  auto ar = view_rows;

  auto ax = DATA_X(0);
  auto ay = DATA_Y(0);
  
  auto bx = DATA_X(ac);
  auto by = DATA_Y(ar);

  auto slice = ROUND(this->slice);
  auto x = ROUND(this->x);
  auto y = ROUND(this->y);

  sprintf(buf, "Image %d / %d", slice+1, slices); os << buf << std::endl;
  sprintf(buf, "Image Size [%d %d]", data_cols, data_rows); os << buf << std::endl;
  sprintf(buf, "View [%d %d] [%d %d]", ax, ay, bx, by); os << buf << std::endl;
  sprintf(buf, "View Size [%d %d]", ac, ar); os << buf << std::endl;
  sprintf(buf, "Channels %d", CHANNELS); os << buf << std::endl;
  sprintf(buf, "Step %d", action_step); os << buf << std::endl;
  sprintf(buf, "Scale %.3f", scale); os << buf << std::endl;
  sprintf(buf, "Position [%d %d]", x, y); os << buf << std::endl;
  sprintf(buf, "Depth [%d]", slice); os << buf << std::endl;

  return os.str();
}

////////////////////////// instance actions ////////////////////////////////

void RLEnv::action_up()
{
  action_step += 1;
  y -= 1.0 / scale;
}

void RLEnv::action_down()
{
  action_step += 1;
  y += 1.0 / scale;
}

void RLEnv::action_left()
{
  action_step += 1;
  x -= 1.0 / scale;
}

void RLEnv::action_right()
{
  action_step += 1;
  x += 1.0 / scale;
}

void RLEnv::action_forward()
{
  action_step += 1;
  // to forward with scaled step uncommend scale in expression below
  slice = std::fmin(slice + 1.0 /*/ scale */, slices-1);
}

void RLEnv::action_backward()
{
  action_step += 1;
  // to backward with scaled step uncommend scale in expression below
  slice = std::fmax(slice - 1.0 /*/ scale */, 0);
}

void RLEnv::action_zoom_in()
{
  action_step += 1;

  float max_scale = MAX_SCALE;

  auto s = scale * 2;

  if (s <= max_scale) scale = s;
}

void RLEnv::action_zoom_out()
{
  action_step += 1;

  float min_s = MIN_SCALE * std::fmin(float(view_rows) / data_rows, float(view_cols) / data_cols);
 
  auto s = scale / 2;

  if (s >= min_s) scale = s;
}

void RLEnv::action_horizontal(float rx)
{
  action_step += 1;

  x += rx * view_cols / scale;
}

void RLEnv::action_vertical(float ry)
{
  action_step += 1;

  y += ry * view_rows / scale;
}

void RLEnv::action_depth(float rz)
{
  action_step += 1;

  slice += rz * slices;
  slice = std::fmax(slice, slices-1);
  slice = std::fmin(slice, 0);
}

void RLEnv::action_zoom(float zoom)
{
  action_step += 1;

  auto s = zoom * scale;
  if (s <= MIN_SCALE) s = MIN_SCALE;
  if (s >= MAX_SCALE) s = MAX_SCALE;
  scale = s;
}

////////////////////////// private methods /////////////////////////////////

uint32_t RLEnv::data_row(uint32_t view_row)
{
  return DATA_Y(view_row);
}

uint32_t RLEnv::data_col(uint32_t view_col)
{
  return DATA_X(view_col);
}

uint32_t RLEnv::view_row(uint32_t data_row)
{
  return DATA_Y(data_row);
}

uint32_t RLEnv::view_col(uint32_t data_col)
{
  return DATA_X(data_col);
}

void RLEnv::clear()
{
  delete[] data;
  data = nullptr;
}

void RLEnv::draw_agent_frame(Image& img)
{
  if (show_view_frame)
  {
    PointVector frame = {
      Point(DATA_X(0)-1, DATA_Y(0)-1),
      Point(DATA_X(0)-1 + DATA_L(view_cols)+1, DATA_Y(0)-1),
      Point(DATA_X(0)-1 + DATA_L(view_cols)+1, DATA_Y(0)-1 + DATA_L(view_rows)+1),
      Point(DATA_X(0)-1, DATA_Y(0)-1 + DATA_L(view_rows)+1),
    };
  
    Painter painter(img.rows(), img.cols());
    painter.draw_polyline(frame, true);
    auto points = painter.output();
    
    for (auto it=points.begin(); it!=points.end(); it++)
    {
      auto& pt = *it;
      img.set(pt.y(), pt.x(), 0xFF, 0xFF, 0x00); // yellow
    }
  }
}

} /* namespace */
