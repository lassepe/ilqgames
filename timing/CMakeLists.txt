# Loop over the executables and create the targets.
file(GLOB timing_srcs ${CMAKE_SOURCE_DIR}/timing/*.cpp)

foreach(timing_src ${timing_srcs})
  get_filename_component(timing_target ${timing_src} NAME_WE)
  message("Including binary \"${BoldMagenta}${timing_target}${ColorReset}\".")
  add_executable(${timing_target} ${timing_src})
  target_link_libraries(${timing_target} ilqgames ${ilqgames_LIBRARIES})
  ilqgames_set_runtime_directory(${timing_target} ${EXECUTABLE_OUTPUT_PATH}/timing)
endforeach()

# Define a variable storing the path to executable directory.
add_definitions(-DILQGAMES_EXEC_DIR="${CMAKE_SOURCE_DIR}/timing")

# Define the path to the log directory.
add_definitions(-DILQGAMES_LOG_DIR="${CMAKE_SOURCE_DIR}/logs")
