# Use the same CANN include paths and C++ ABI as the upstream host libraries.
add_library(paretopilot_onnx SHARED ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp)
target_include_directories(paretopilot_onnx PRIVATE ${METADEF_INCLUDE_DIRS} ${OPBASE_INC_DIRS})
target_link_directories(paretopilot_onnx PRIVATE ${ASCEND_DIR}/lib64 ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
target_link_libraries(paretopilot_onnx PRIVATE intf_pub_cxx17 register graph)
set_target_properties(paretopilot_onnx PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
