//================================================================
//  Common Log Display
// (c) qiu.chao ,2016
//================================================================
#ifndef HLS_CM_LOG_H
#define HLS_CM_LOG_H

// ================================
// = Standard C Libraries =
// ================================
#include <cstdio> // printf
//#include <hls_video.h>

//====Log Display

extern int LOG_LEVEL;
extern bool LOG_DETAILS;

#define LOG_LEVEL_INCR LOG_LEVEL++
#define LOG_LEVEL_DECR                                                         \
  if (LOG_LEVEL > 0)                                                           \
  LOG_LEVEL--

template <typename TYPE> void print_indent(TYPE lvl) {
  while (lvl--) {
    putchar(' ');
    putchar(' ');
  }
}

#define LOG_RESULT(...)                                                        \
  {                                                                            \
    print_indent<int>(LOG_LEVEL);                                              \
    printf(__VA_ARGS__);                                                       \
  }

//#define LOG_DEBUG

#if defined(LOG_DEBUG)
#define LOG(...)                                                               \
  {                                                                            \
    print_indent<int>(LOG_LEVEL);                                              \
    printf(__VA_ARGS__);                                                       \
  }
#else
#define LOG(...)                                                               \
  {}
#endif

//==== usage methord
// LOG("The number is %d",num);

#endif /* HLS_LOG_HPP */
