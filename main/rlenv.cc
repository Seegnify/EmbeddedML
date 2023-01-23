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
#define FULL_X(ax) ROUND((ax) / this->scale + (OFFSET_X()))
#define FULL_Y(ay) ROUND((ay) / this->scale + (OFFSET_Y()))
#define FULL_L(al) ROUND((al) / this->scale) // data length

// view dimensions
#define VIEW_X(rx) ROUND((rx - (OFFSET_X())) * this->scale)
#define VIEW_Y(ry) ROUND((ry - (OFFSET_Y())) * this->scale)
#define VIEW_L(rl) ROUND((rl) * this->scale) // view length

// view/data array index
#define VIEW_I(x,y,channel) (3*((y)*this->view_cols + (x)) + (channel))
#define DATA_I(x,y,channel) (3*((y)*this->full_cols + (x)) + (channel))

// frame color
#define FRAME_RGB 0x00FFFF

RLEnv::RLEnv()
{
  data = nullptr;

  show_full_frame = false;
  show_view_frame = false;

  slices = 0;
  full_rows = 0;
  full_cols = 0;
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
  x = full_cols / 2;
  y = full_rows / 2;
}

void RLEnv::set_full_rgb(
  const uint8_t* rgb,
  uint16_t slices, uint16_t rows, uint16_t cols
)
{
  // clear data if needed
  if (this->slices != slices || 
      this->full_rows != rows ||
      this->full_cols != cols)
    clear();

  // allocated data if needed  
  int size = CHANNELS * slices * rows * cols;
  if (this->data == nullptr) this->data = new uint8_t[size];
  memcpy(this->data, rgb, size);

  // update dimenstions
  this->slices = slices;
  this->full_rows = rows;
  this->full_cols = cols;
}

void RLEnv::set_view_size(uint16_t rows, uint16_t cols)
{
  this->view_rows = rows;
  this->view_cols = cols;
}

////////////////////////// instance UI API /////////////////////////////////

void RLEnv::get_full_size(uint16_t& full_rows, uint16_t& full_cols)
{
  full_rows = this->full_rows;
  full_cols = this->full_cols;
}

void RLEnv::get_view_size(uint16_t& view_rows, uint16_t& view_cols)
{
  view_rows = this->view_rows;
  view_cols = this->view_cols;
}

Image RLEnv::get_full_rgb()
{
  // get slice index
  auto slice = ROUND(this->slice);

  // copy full data
  int slice_offset = CHANNELS * slice * full_rows * full_cols;
  Image full(full_rows, full_cols, CHANNELS);
  memcpy(full.data(), data + slice_offset, full.size());
  
  // draw view frame
  draw_view_frame(full);

  return full;
}

Image RLEnv::get_view_rgb()
{
  // crop data to agent view
  Image full = get_full_rgb();
  auto view = full.crop(FULL_Y(0), FULL_X(0), FULL_L(view_rows), FULL_L(view_cols));

  // zoom to fit agent view
  view = view.scale(view_rows, view_cols);

  // draw full frame
  draw_full_frame(view);

  return view;
}

std::string RLEnv::get_info()
{
  char buf[128];
  std::ostringstream os;

  auto ac = view_cols;
  auto ar = view_rows;

  auto ax = FULL_X(0);
  auto ay = FULL_Y(0);
  
  auto bx = FULL_X(ac);
  auto by = FULL_Y(ar);

  auto slice = ROUND(this->slice);
  auto x = ROUND(this->x);
  auto y = ROUND(this->y);

  sprintf(buf, "Image %d / %d", slice+1, slices); os << buf << std::endl;
  sprintf(buf, "Image Size [%d %d]", full_cols, full_rows); os << buf << std::endl;
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

  float min_s = MIN_SCALE * std::fmin(float(view_rows) / full_rows, float(view_cols) / full_cols);
 
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
  return FULL_Y(view_row);
}

uint32_t RLEnv::data_col(uint32_t view_col)
{
  return FULL_X(view_col);
}

uint32_t RLEnv::view_row(uint32_t full_row)
{
  return VIEW_Y(full_row);
}

uint32_t RLEnv::view_col(uint32_t full_col)
{
  return VIEW_X(full_col);
}

void RLEnv::clear()
{
  delete[] data;
  data = nullptr;
}

void RLEnv::draw_full_frame(Image& img)
{
  if (show_full_frame)
  {
    PointVector frame = {
      Point(VIEW_X(0)-1, VIEW_Y(0)-1),
      Point(VIEW_X(0)-1 + VIEW_L(full_cols)+1, VIEW_Y(0)-1),
      Point(VIEW_X(0)-1 + VIEW_L(full_cols)+1, VIEW_Y(0)-1 + VIEW_L(full_rows)+1),
      Point(VIEW_X(0)-1, VIEW_Y(0)-1 + VIEW_L(full_rows)+1),
    };

    Painter painter(img.rows(), img.cols());
    painter.draw_polyline(frame, true);
    auto points = painter.output();

    for (auto it=points.begin(); it!=points.end(); it++)
    {
      auto& pt = *it;
      img.set(pt.y(), pt.x(),
        (FRAME_RGB & 0xFF0000) >> 16,
        (FRAME_RGB & 0x00FF00) >> 8,
        (FRAME_RGB & 0x0000FF)
      );
    }
  }
}

void RLEnv::draw_view_frame(Image& img)
{
  if (show_view_frame)
  {
    PointVector frame = {
      Point(FULL_X(0)-1, FULL_Y(0)-1),
      Point(FULL_X(0)-1 + FULL_L(view_cols)+1, FULL_Y(0)-1),
      Point(FULL_X(0)-1 + FULL_L(view_cols)+1, FULL_Y(0)-1 + FULL_L(view_rows)+1),
      Point(FULL_X(0)-1, FULL_Y(0)-1 + FULL_L(view_rows)+1),
    };

    Painter painter(img.rows(), img.cols());
    painter.draw_polyline(frame, true);
    auto points = painter.output();

    for (auto it=points.begin(); it!=points.end(); it++)
    {
      auto& pt = *it;
      img.set(pt.y(), pt.x(),
        (FRAME_RGB & 0xFF0000) >> 16,
        (FRAME_RGB & 0x00FF00) >> 8,
        (FRAME_RGB & 0x0000FF)
      );
    }
  }
}

} /* namespace */
