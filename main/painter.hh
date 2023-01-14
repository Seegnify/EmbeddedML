#ifndef _SEEGNIFY_PAINTER_H_
#define _SEEGNIFY_PAINTER_H_

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

namespace seegnify {

struct Point
{
public:
  Point(int x, int y)
  {
    _x = x;
    _y = y;
  }

  inline int x() const { return _x; }
  inline int y() const { return _y; }

  bool operator==(const Point& other) const
  {
    return _x == other._x && _y == other._y;
  }

protected:
  int _x;
  int _y;
};

struct PointHash
{
  size_t operator()(const Point& p) const
  {
    return ((uint32_t)p.x()) << 16 + p.y();
  }
};

typedef std::unordered_set<Point, PointHash> PointSet;
typedef std::vector<Point> PointVector;

class Painter
{
public:
  Painter(int rows=-1, int cols=-1)
  {
    _rows = rows;
    _cols = cols;
  }

  int rows() { return _rows; }

  int cols() { return _cols; }

  const PointSet& output() const
  {
    return _output;
  }

  void draw_polygon(const PointVector& polygon)
  {
    int size = polygon.size();
    if (size > 0)
    {
      // draw outline
      draw_polyline(polygon);
      auto& first = polygon.front();
      auto& last = polygon.back();
      draw_line(first.x(), first.y(), last.x(), last.y());

      // collect lines: y->{x0,...,xn}
      std::unordered_map<int, std::unordered_set<int>> lines;
      for (auto& p: _output)
      {
        auto line = lines.find(p.y());
        if (line == lines.end())
        {
          lines.emplace(p.y(), std::unordered_set<int>());
          line = lines.find(p.y());
        }
        line->second.insert(p.x());
      }

      // fill points between edges
      for (auto it=lines.begin(); it!=lines.end(); it++)
      {
        auto y = it->first;
        std::vector<int> x_list(it->second.begin(), it->second.end()); 
        std::sort(x_list.begin(), x_list.end());
        auto x_it = x_list.begin();
        auto x = *x_it;
        bool fill = true;
        for (x_it++; x_it!=x_list.end(); x_it++)
        {
          // draw line from previus x to current x
          if (fill) draw_line(x, y, *x_it, y);
          if (abs(*x_it-x) > 1) fill = !fill;
          x = *x_it;
        }
      }
    }
  }

  void draw_polyline(const PointVector& polygon, bool close=false)
  {
    int size = polygon.size();
    for (int i=1; i<size; i++) draw_line(polygon[i-1], polygon[i]);
    if (close && size > 1) draw_line(polygon.back(), polygon.front());
  }

  void draw_line(const Point& a, const Point& b)
  {
    draw_line(a.x(), a.y(), b.x(), b.y());
  }

  void draw_line(int x0, int y0, int x1, int y1)
  {
    // Bresenham's line algorithm
    int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
    int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1; 
    int err = (dx>dy ? dx : -dy)/2, e2;
   
    for(;;){
      draw_point(x0, y0);
      if (x0==x1 && y0==y1) break;
      e2 = err;
      if (e2 >-dx) { err -= dy; x0 += sx; }
      if (e2 < dy) { err += dx; y0 += sy; }
    }
  }

  void draw_point(const Point& p)
  {
    draw_point(p.x(), p.y());
  }

  void draw_point(int x, int y)
  {
    if ((_cols < 0 || _rows < 0) || (x >= 0 && x < _cols && y >= 0 && y < _rows))
    {
      _output.insert(Point(x, y));
    }
  }

private:
  int _rows;
  int _cols;
  PointSet _output;
};

} /* namespace */

#endif /* _SEEGNIFY_PAINTER_H_ */
