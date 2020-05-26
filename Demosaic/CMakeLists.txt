# https://stackoverflow.com/questions/13703647/how-to-properly-add-include-directories-with-cmake
cmake_minimum_required(VERSION 2.8)
project( Demosaicing )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Demosaicing Demosaicing.cpp GBTF.cpp)
target_link_libraries( Demosaicing ${OpenCV_LIBS} )

